"""
environment.py — AutoFactoryToDEnv

A 24-step (hourly) reinforcement-learning environment for factory scheduling.

Objectives
----------
* Meet 8 000 parts/day target
* Minimise electricity cost (time-of-day tariffs)
* Reduce machine wear
* Reduce CO₂ emissions

Action space (MultiDiscrete)
----------------------------
  stamping   : 0 idle | 1 half | 2 full
  molding    : 0 idle | 1 half | 2 full
  cnc        : 0 idle | 1 half | 2 full
  compressor : 0 off  | 1 on   (boosts total output +15 %)
  welder     : 0 off  | 1 full | 2 maintenance

Observation
-----------
  hour, production_so_far, machine_health[5], electricity_price
"""

from __future__ import annotations

import random
from typing import Any, Dict, List, Tuple


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TOTAL_HOURS = 24
PRODUCTION_TARGET = 8_000      # parts per day

# Worst-case reference values for score normalisation (24 hrs, all full + compressor)
MAX_EPISODE_COST   = 24 * 23.0          # ~$552  (all machines full, peak tariff)
MAX_EPISODE_CO2    = 24 * 2.85 * 0.4    # ~27.4 kg
MAX_EPISODE_ENERGY = 24 * 2.85          # ~68.4 MWh

# Base parts produced per machine per hour at each power level
#                    idle  half  full
MACHINE_OUTPUT = {
    "stamping":   [0,  150,  300],
    "molding":    [0,  130,  260],
    "cnc":        [0,  110,  220],
}

COMPRESSOR_BOOST = 1.15        # +15% on total production when on
WELDER_OUTPUT    = [0, 80, 0]  # idle / full / maintenance (no parts during maintenance)

# Electricity tariffs ($/MWh)
TARIFF_PEAK   = 9.85   # hours 6–10, 18–22
TARIFF_NORMAL = 7.88   # hours 5–6,  10–18
TARIFF_NIGHT  = 7.48   # hours 0–5,  22–24

# Power draw (MWh) per machine per hour at each level
#                            idle  half  full
MACHINE_POWER = {
    "stamping":   [0.0, 0.4, 0.8],
    "molding":    [0.0, 0.35, 0.70],
    "cnc":        [0.0, 0.3, 0.60],
    "compressor": [0.0, 0.25],        # off / on
    "welder":     [0.0, 0.5, 0.1],   # off / full / maintenance
}

# Health loss per hour at each level (0 = idle, full degrades faster)
#                             idle   half   full
HEALTH_LOSS = {
    "stamping":   [0.0, 0.005, 0.012],
    "molding":    [0.0, 0.005, 0.012],
    "cnc":        [0.0, 0.004, 0.010],
    "compressor": [0.0, 0.008],
    "welder":     [0.0, 0.008, -0.020],  # maintenance restores health
}

# CO₂ intensity (kg per MWh)
CO2_INTENSITY = 0.4

# Machine breakdown threshold — below this health, machine is "broken"
BREAKDOWN_THRESHOLD = 0.15

W_PROD       =  0.08   # hourly production progress toward target
W_COST       =  0.05   # electricity cost penalty
W_WEAR       =  0.05   # machine health degradation penalty
W_CO2        =  0.02   # carbon emissions penalty
W_BREAKDOWN  =  0.15   # hard penalty per broken machine
W_TERMINAL   =  0.08   # end-of-episode bonus/shortfall penalty

# Normalisation reference values  (approximate worst-case per step)
MAX_HOURLY_PRODUCTION = 989.0   # all full + compressor  (780*1.15 + 80*1.15)
MAX_HOURLY_COST       = 23.0    # all full at peak tariff (2.35 MWh * 9.85)
MAX_HOURLY_CO2        = 1.0     # 2.35 MWh * 0.4


# ---------------------------------------------------------------------------
# Helper: electricity tariff
# ---------------------------------------------------------------------------

def get_tariff(hour: int) -> float:
    """
    Return the electricity tariff ($/MWh) for the given hour (0–23).

    Peak   (9.85): hours 6–10, 18–22
    Normal (7.88): hours 5–6,  10–18
    Night  (7.48): hours 0–5,  22–24
    """
    if   6  <= hour < 10: return TARIFF_PEAK
    elif 18 <= hour < 22: return TARIFF_PEAK
    elif 5  <= hour < 6:  return TARIFF_NORMAL
    elif 10 <= hour < 18: return TARIFF_NORMAL
    else:                 return TARIFF_NIGHT   # 0–5 and 22–24


def is_peak_hour(hour: int) -> bool:
    """Return True if the given hour falls within a peak tariff window."""
    return get_tariff(hour) == TARIFF_PEAK


# ---------------------------------------------------------------------------
# Helper: total power draw (shared by cost and CO₂)
# ---------------------------------------------------------------------------

def compute_total_power(
    stamping:   int,
    molding:    int,
    cnc:        int,
    compressor: int,
    welder:     int,
) -> float:
    """Return total power draw (MWh) for all machines in one hour."""
    return (
        MACHINE_POWER["stamping"][stamping]
        + MACHINE_POWER["molding"][molding]
        + MACHINE_POWER["cnc"][cnc]
        + MACHINE_POWER["compressor"][compressor]
        + MACHINE_POWER["welder"][welder]
    )


# ---------------------------------------------------------------------------
# Helper: breakdown probability
# ---------------------------------------------------------------------------

def breakdown_probability(health: float) -> float:
    """
    Return the probability [0, 1] that a machine suffers a breakdown
    (produces zero output) this hour.

    Probability rises sharply as health falls below 0.40:
      health >= 0.40 → 0 % chance
      health == 0.20 → ~10 % chance
      health == 0.10 → ~30 % chance
      health == 0.00 → 100 % chance

    Formula: max(0, (0.40 - health) / 0.40) ** 1.5
    """
    if health >= 0.40:
        return 0.0
    return ((0.40 - health) / 0.40) ** 1.5


# ---------------------------------------------------------------------------
# Helper: production this step
# ---------------------------------------------------------------------------

def compute_production(
    stamping:       int,
    molding:        int,
    cnc:            int,
    compressor:     int,
    welder:         int,
    machine_health: List[float],
) -> tuple[float, List[bool]]:
    """
    Return (parts_produced, breakdown_flags) for one hour.

    Production is scaled by each machine's current health — a machine at
    50 % health produces 50 % of its rated output.  Additionally, each
    machine has a stochastic breakdown chance that rises as health falls
    below 0.40; a broken-down machine contributes zero output this step.

    breakdown_flags : list[bool] in machine order
                      [stamping, molding, cnc, compressor, welder]
                      True means the machine broke down this step.
    """
    # Per-machine: (level,  base_output,                 health_index)
    machines = [
        (stamping,   MACHINE_OUTPUT["stamping"][stamping],   0),
        (molding,    MACHINE_OUTPUT["molding"][molding],     1),
        (cnc,        MACHINE_OUTPUT["cnc"][cnc],             2),
        (0,          0.0,                                    3),   # compressor: no direct output
        (welder,     float(WELDER_OUTPUT[welder]),           4),
    ]

    output        = 0.0
    breakdown_flags = [False] * 5

    for level, base, idx in machines:
        if level == 0:          # idle / off — no output, no breakdown risk
            continue
        health = machine_health[idx]
        if random.random() < breakdown_probability(health):
            breakdown_flags[idx] = True
            continue            # machine failed this step — no output
        # Scale output by health (e.g. 0.7 health → 70 % of rated output)
        output += base * health

    # Compressor boosts total output when on and not broken down
    if compressor == 1 and not breakdown_flags[3]:
        output *= COMPRESSOR_BOOST

    return output, breakdown_flags


# ---------------------------------------------------------------------------
# Helper: electricity cost this step
# ---------------------------------------------------------------------------

def compute_cost(
    stamping:   int,
    molding:    int,
    cnc:        int,
    compressor: int,
    welder:     int,
    hour:       int,
) -> float:
    """Return electricity cost ($) incurred in one hour."""
    power  = compute_total_power(stamping, molding, cnc, compressor, welder)
    tariff = get_tariff(hour)
    return power * tariff


# ---------------------------------------------------------------------------
# Helper: health updates this step
# ---------------------------------------------------------------------------

def compute_health_delta(
    stamping:   int,
    molding:    int,
    cnc:        int,
    compressor: int,
    welder:     int,
) -> List[float]:
    """
    Return per-machine health delta for one hour.
    Negative values mean degradation; positive values mean recovery.
    Order: [stamping, molding, cnc, compressor, welder]

    Welder in maintenance mode (level 2) restores health (+0.020/hr).
    """
    return [
        -HEALTH_LOSS["stamping"][stamping],
        -HEALTH_LOSS["molding"][molding],
        -HEALTH_LOSS["cnc"][cnc],
        -HEALTH_LOSS["compressor"][compressor],
        -HEALTH_LOSS["welder"][welder],
    ]


# ---------------------------------------------------------------------------
# Helper: CO₂ this step
# ---------------------------------------------------------------------------

def compute_co2(
    stamping:   int,
    molding:    int,
    cnc:        int,
    compressor: int,
    welder:     int,
) -> float:
    """Return CO₂ emissions (kg) for one hour."""
    power = compute_total_power(stamping, molding, cnc, compressor, welder)
    return power * CO2_INTENSITY


# ---------------------------------------------------------------------------
# Helper: reward  (V5 — hackathon-compliant, strictly [0, 1])
# ---------------------------------------------------------------------------

def compute_step_reward(
    production_delta:   float,
    cost:               float,
    health_delta:       List[float],
    co2:                float,
    machine_health:     List[float],
    hour:               int,
    is_terminal:        bool,
    production_so_far:  float,
    production_target:  float = PRODUCTION_TARGET,  # TASK 3: dynamic target
    hours_over_cap:     int = 0,
) -> float:
    """
    Hackathon-compliant step reward, strictly bounded in [0.0, 1.0].

    Weights
    -------
      0.40  production progress  (cumulative / target)
      0.20  machine health       (avg health across 5 machines)
      0.20  cost efficiency      (1 − step_cost / max_hourly_cost)
      0.20  CO₂ efficiency       (1 − step_co2 / max_hourly_co2)

    Bonuses / Penalties
    -------------------
      −0.05  idling before target is reached (production_delta == 0)
      −0.05  running full power during peak hours
      +0.10  reaching cumulative target (applied once per terminal)
    """
    # --- component scores (each independently in [0, 1]) ---
    # TASK 3: use dynamic target production
    progress    = min(production_so_far / production_target, 1.0)              # 0→1
    cost_norm   = min(cost / MAX_HOURLY_COST, 1.0)                           # 0→1 (bad=1)
    co2_norm    = min(co2 / MAX_HOURLY_CO2, 1.0)                             # 0→1 (bad=1)
    health_norm = sum(machine_health) / len(machine_health)                  # 0→1

    reward = (
        0.40 * progress
        + 0.20 * health_norm
        + 0.20 * (1.0 - cost_norm)
        + 0.20 * (1.0 - co2_norm)
    )

    # --- penalties ---
    # Idle penalty: no production while target unmet wastes hours
    if production_delta == 0.0 and production_so_far < production_target:
        reward -= 0.05

    # Peak penalty: running at full power during expensive peak hours
    all_full = (cost_norm > 0.85)   # proxy for "high power" action
    if is_peak_hour(hour) and all_full:
        reward -= 0.05

    # Smooth production enforcement: penalise if >20% ahead of schedule
    # Expected: produce evenly across 24 hours
    expected_progress = (hour / 24.0) * production_target
    if expected_progress > 0 and production_so_far > expected_progress * 1.2:
        reward -= 0.05

    # --- terminal bonus ---
    if is_terminal and production_so_far >= production_target:
        reward += 0.10

    return round(float(max(0.0, min(1.0, reward))), 6)


# Keep the old name as an alias so existing callers don't break.
compute_reward = compute_step_reward


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

class AutoFactoryToDEnv:
    """
    AutoFactoryToDEnv — factory production scheduling RL environment.

    Usage
    -----
    env = AutoFactoryToDEnv()
    obs, info = env.reset()
    obs, reward, terminated, truncated, info = env.step(action)
    """

    # machine names in health list order
    MACHINE_NAMES = ["stamping", "molding", "cnc", "compressor", "welder"]

    def __init__(
        self,
        task:               str   = "medium",  # TASK 1: new task mode param
        target:             int | None = None, # optional override
        fixed_tariff:       float | None = None,
        enable_breakdowns:  bool  = True,
        production_noise:   bool  = True,
    ) -> None:
        # --- TASK CONFIG (TASK 1 & 2) ---
        self.task_mode = task.lower()
        if self.task_mode == "easy":
            self.target_production = 6000
            self.breakdown_prob    = 0.0
            self.use_time_of_day   = False
        elif self.task_mode == "medium":
            self.target_production = 8000
            self.breakdown_prob    = 0.05
            self.use_time_of_day   = True
        elif self.task_mode == "hard":
            self.target_production = 8000
            self.breakdown_prob    = 0.10
            self.use_time_of_day   = True
        else:
            raise ValueError(f"Unknown task mode: {task}. Choose easy, medium, or hard.")

        # Override defaults if explicitly provided
        if target is not None: self.target_production = target
        
        # TASK 4 & 6: Hard mode extras
        self.rush_order_hour   = 12 if self.task_mode == "hard" else None
        self.rush_order_extra  = 1500 if self.task_mode == "hard" else 0
        self.maintenance_hours = [8, 18] if self.task_mode == "hard" else []

        # --- remaining config ---
        self.fixed_tariff:       float | None = fixed_tariff
        self.enable_breakdowns:  bool       = enable_breakdowns
        self.production_noise:   bool       = production_noise

        # TASK 9 — print task mode at the top of output (Once per env)
        print(f"TASK MODE: {self.task_mode.upper()}")

        # --- episode state ---
        self.hour:              int        = 0
        self.production_so_far: float      = 0.0
        self.machine_health:    List[float] = [1.0] * 5
        self._terminated:       bool       = False

        # --- cumulative tracking (for state() / compute_score) ---
        self.total_cost:        float      = 0.0
        self.total_co2:         float      = 0.0
        self.total_energy:      float      = 0.0
        self.peak_energy:       float      = 0.0
        self.breakdown_log:     List[Dict[str, Any]] = []
        
        # --- internal mechanics tracking ---
        self.hours_over_cap:    int        = 0

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_tariff(self, hour: int) -> float:
        """Return tariff, respecting fixed_tariff and task config."""
        if self.fixed_tariff is not None:
            return self.fixed_tariff
        # TASK 7: flat rate for easy mode
        if not self.use_time_of_day:
            return 750.0
        return get_tariff(hour)

    def _breakdown_prob(self, health: float) -> float:
        """Return breakdown probability, respecting task config."""
        # TASK 5: use task-specific breakdown prob
        if not self.enable_breakdowns:
            return 0.0
        return self.breakdown_prob

    def _compute_production(
        self,
        stamping:   int,
        molding:    int,
        cnc:        int,
        compressor: int,
        welder:     int,
    ) -> tuple[float, List[bool]]:
        """Instance-level production: uses self._breakdown_prob for config awareness."""
        machines = [
            (stamping,   MACHINE_OUTPUT["stamping"][stamping],   0),
            (molding,    MACHINE_OUTPUT["molding"][molding],     1),
            (cnc,        MACHINE_OUTPUT["cnc"][cnc],             2),
            (0,          0.0,                                    3),
            (welder,     float(WELDER_OUTPUT[welder]),           4),
        ]
        output        = 0.0
        breakdown_flags = [False] * 5

        for level, base, idx in machines:
            if level == 0:
                continue
            health = self.machine_health[idx]
            if random.random() < self._breakdown_prob(health):
                breakdown_flags[idx] = True
                continue
            output += base * health

        if compressor == 1 and not breakdown_flags[3]:
            output *= COMPRESSOR_BOOST

        # TASK 2 — stronger stochastic noise: ±15% Gaussian jitter
        if self.production_noise and output > 0:
            noise = random.gauss(0.0, 0.15)          # N(0, σ=0.15)  ← was 0.05
            noise = max(noise, -0.50)                 # floor: never lose >50%
            output = max(0.0, output * (1.0 + noise))

            # Unconditional 10% partial-breakdown chance (policy must be robust)
            if random.random() < 0.10:
                output *= 0.5

        return output, breakdown_flags

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Reset environment state and return initial observation."""
        # TASK 8 — reset core state
        self.hour              = 0
        self.production_so_far = 0.0
        self._terminated       = False
        
        # Reset task target (in case it was modified by rush order)
        if self.task_mode == "hard":
            self.target_production = 8000
            
        self.machine_health    = [1.0] * 5

        # Reset cumulative tracking
        self.total_cost        = 0.0
        self.total_co2         = 0.0
        self.total_energy      = 0.0
        self.peak_energy       = 0.0
        self.breakdown_log     = []
        self.hours_over_cap    = 0

        obs  = self._build_observation()
        info = {"message": "Episode reset. Good luck!"}
        return obs, info

    def step(
        self,
        stamping:   int,
        molding:    int,
        cnc:        int,
        compressor: int,
        welder:     int,
    ) -> Tuple[Dict[str, Any], float, bool, bool, Dict[str, Any]]:
        """
        Advance environment by one hour.

        Parameters
        ----------
        stamping, molding, cnc : 0=idle, 1=half, 2=full
        compressor : 0=off, 1=on
        welder     : 0=off, 1=full, 2=maintenance

        Returns
        -------
        observation, reward, terminated, truncated, info
        """
        if self._terminated:
            raise RuntimeError("Episode is over. Call reset() first.")

        # TASK 4 — Rush order logic (Hard mode)
        if self.rush_order_hour is not None and self.hour == self.rush_order_hour:
            self.target_production += self.rush_order_extra

        # TASK 6 — Forced maintenance logic (Hard mode)
        if self.maintenance_hours and self.hour in self.maintenance_hours:
            # Override welding machine to maintenance mode (2) or off (0)?
            # User snippet says action[-1] = 0, so we use 0 (off).
            welder = 0

        # --- production (health-scaled, with stochastic breakdowns) ---
        # Use env-level config to control breakdowns
        production_delta, breakdown_flags = self._compute_production(
            stamping, molding, cnc, compressor, welder,
        )

        # --- cost (uses current-hour tariff, respecting fixed_tariff) ---
        power  = compute_total_power(stamping, molding, cnc, compressor, welder)
        tariff = self._get_tariff(self.hour)
        cost   = power * tariff

        # --- health delta (degradation or maintenance recovery) ---
        health_delta = compute_health_delta(
            stamping, molding, cnc, compressor, welder
        )

        # --- CO₂ (proportional to power drawn, not production) ---
        co2 = power * CO2_INTENSITY

        # --- Hard cap: once cumulative production >= target,
        #     zero out ALL step outputs so the agent is not rewarded
        #     for burning energy after the goal is complete.
        if self.production_so_far >= self.target_production:
            self.hours_over_cap += 1
            production_delta = 0.0   # no parts
            cost             = 0.0   # no cost charged (machine idling is free)
            co2              = 0.0   # no emissions
            # Skip energy/cost/co2 accumulation for this step
        else:
            # --- accumulate tracking (only while target unmet) ---
            self.total_cost   += cost
            self.total_co2    += co2
            self.total_energy += power
            if is_peak_hour(self.hour):
                self.peak_energy += power

            # Clamp partial step so we never exceed target
            if self.production_so_far + production_delta > self.target_production:
                production_delta = self.target_production - self.production_so_far
            self.production_so_far += production_delta

        # Guaranteed hard cap (defensive)
        self.production_so_far = min(self.production_so_far, self.target_production)

        self.machine_health = [
            max(0.0, min(1.0, h + d))
            for h, d in zip(self.machine_health, health_delta)
        ]
        self.hour += 1

        # --- termination ---
        terminated = self.hour >= TOTAL_HOURS
        self._terminated = terminated

        # --- reward (computed AFTER state update so reward sees new health) ---
        reward = compute_step_reward(
            production_delta  = production_delta,
            cost              = cost,
            health_delta      = health_delta,
            co2               = co2,
            machine_health    = self.machine_health,
            hour              = self.hour - 1,   # the hour we just acted in
            is_terminal       = terminated,
            production_so_far = self.production_so_far,
            production_target = self.target_production, # TASK 3
            hours_over_cap    = self.hours_over_cap,
        )

        obs  = self._build_observation()
        info = {
            "production_delta":  round(production_delta, 2),
            "cost_usd":          round(cost, 4),
            "co2_kg":            round(co2, 4),
            "health_delta":      [round(d, 4) for d in health_delta],
            "actual_action":     [stamping, molding, cnc, compressor, welder],
            "breakdown_events":  {
                name: broke
                for name, broke in zip(AutoFactoryToDEnv.MACHINE_NAMES, breakdown_flags)
                if broke
            },
            "target_met":        self.production_so_far >= self.target_production if terminated else None,
        }

        # TASK 4 — include final_score in info at episode end
        if terminated:
            info["final_score"] = compute_final_score(self.state())

        # Log breakdown events
        bd_dict = {
            name: True
            for name, broke in zip(AutoFactoryToDEnv.MACHINE_NAMES, breakdown_flags)
            if broke
        }
        if bd_dict:
            self.breakdown_log.append({"hour": self.hour - 1, "machines": bd_dict})

        return obs, reward, terminated, False, info

    # ------------------------------------------------------------------
    # state() — full internal state for compute_score / evaluation
    # ------------------------------------------------------------------

    def state(self) -> Dict[str, Any]:
        """
        Return the full internal state of the environment.

        This is used by ``compute_score()`` and ``evaluate_policy()``.
        It is NOT the observation — it contains privileged information
        that includes cumulative cost, CO₂, energy, and breakdown log.
        """
        return {
            "hour":              self.hour,
            "total_production":  round(self.production_so_far, 2),
            "production_target": self.target_production, # TASK 3
            "machine_health":    [round(h, 4) for h in self.machine_health],
            "total_cost":        round(self.total_cost, 4),
            "total_co2":         round(self.total_co2, 4),
            "total_energy":      round(self.total_energy, 4),
            "peak_energy":       round(self.peak_energy, 4),
            "breakdown_events":  list(self.breakdown_log),
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_observation(self) -> Dict[str, Any]:
        """
        Build the observation dict for the current state.

        `electricity_price` reflects the tariff for the NEXT hour the agent
        will act in (self.hour), so it can plan accordingly.  On the terminal
        step the episode is over, but we still return the last valid tariff.
        """
        next_hour = min(self.hour, TOTAL_HOURS - 1)
        return {
            "hour":              self.hour,
            "production_so_far": round(self.production_so_far, 2),
            "production_target": self.target_production, # TASK 3
            "machine_health":    [round(h, 4) for h in self.machine_health],
            "electricity_price": self._get_tariff(next_hour),
        }


# ==========================================================================
# compute_final_score(state) — TASK 3 — hackathon-compliant, in [0, 1]
# ==========================================================================

def compute_final_score(
    state: Dict[str, Any],
    target: float = PRODUCTION_TARGET,
) -> float:
    """
    Deterministic post-episode evaluation score in [0, 1].

    Weights (must sum to 1.0)
    -------------------------
      0.50  production_score  = cumulative_production / target
      0.20  cost_score        = 1 − (total_cost / 50 000)
      0.20  co2_score         = 1 − (total_co2 / 30.0 kg)
      0.10  health_score      = avg machine health [0, 1]
    """
    production_score = min(state["total_production"] / target, 1.0)

    cost_score  = max(1.0 - (state["total_cost"]  / 50_000.0), 0.0)
    co2_score   = max(1.0 - (state["total_co2"]   / 30.0),     0.0)

    avg_health   = sum(state["machine_health"]) / len(state["machine_health"])
    health_score = avg_health   # already [0, 1]

    score = (
        0.50 * production_score
        + 0.20 * cost_score
        + 0.20 * co2_score
        + 0.10 * health_score
    )
    return round(float(max(0.0, min(1.0, score))), 6)


# Keep old name as an alias (backward-compatible with evaluate_policy, API, etc.)
compute_score = compute_final_score


# ==========================================================================
# Task configs — EASY / MEDIUM / HARD
# ==========================================================================

TASK_CONFIGS: Dict[str, Dict[str, Any]] = {
    "easy": {
        "description":        "Fixed tariff, no breakdowns, relaxed target.",
        "target":              7_000,
        "fixed_tariff":        TARIFF_NORMAL,     # constant 7.88 $/MWh
        "enable_breakdowns":   False,
        "rush_order":          None,
        "forced_maintenance":  None,
        "seed":                1001,
    },
    "medium": {
        "description":        "Time-of-day tariffs, standard target.",
        "target":              8_000,
        "fixed_tariff":        None,               # use ToD schedule
        "enable_breakdowns":   False,
        "rush_order":          None,
        "forced_maintenance":  None,
        "seed":                2002,
    },
    "hard": {
        "description":        "ToD tariffs, breakdowns, rush order at hour 12 (+1500), forced maintenance.",
        "target":              8_000,
        "fixed_tariff":        None,
        "enable_breakdowns":   True,
        "rush_order":          {"hour": 12, "extra_parts": 1500},
        "forced_maintenance":  {"hour": 8, "machine_idx": 4},  # welder
        "seed":                3003,
    },
}


# ==========================================================================
# evaluate_policy(policy, task_name)
# ==========================================================================

def evaluate_policy(
    policy,
    task_name: str = "medium",
) -> Dict[str, Any]:
    """
    Run a full episode using the given policy under a task config.

    Parameters
    ----------
    policy : callable(observation_dict) -> tuple[int,int,int,int,int]
        A function that receives an observation dict and returns an action
        tuple: (stamping, molding, cnc, compressor, welder).
    task_name : str
        One of 'easy', 'medium', 'hard'.

    Returns
    -------
    dict with keys:
        score        : float  — deterministic score [0,1]
        total_reward : float
        task         : str
        metrics      : dict   — full state from env.state()
    """
    if task_name not in TASK_CONFIGS:
        raise ValueError(f"Unknown task: {task_name!r}. Choose from {list(TASK_CONFIGS)}")

    cfg = TASK_CONFIGS[task_name]
    random.seed(cfg["seed"])       # deterministic episode

    # --- create env with task-specific config ---
    env = AutoFactoryToDEnv(
        target            = cfg["target"],
        fixed_tariff      = cfg["fixed_tariff"],
        enable_breakdowns = cfg["enable_breakdowns"],
    )

    obs, info = env.reset()
    total_reward = 0.0

    for h in range(TOTAL_HOURS):
        # --- forced maintenance override ---
        fm = cfg["forced_maintenance"]
        if fm and h == fm["hour"]:
            action = list(policy(obs))
            action[fm["machine_idx"]] = 2  # force maintenance
            action = tuple(action)
        else:
            action = policy(obs)

        obs, reward, terminated, truncated, step_info = env.step(*action)
        total_reward += reward

        # --- rush order: inject extra production at the specified hour ---
        rush = cfg["rush_order"]
        if rush and h == rush["hour"]:
            env.production_so_far = min(
                env.production_so_far + rush["extra_parts"],
                cfg["target"],
            )

        if terminated:
            break

    final_state = env.state()
    score = compute_score(final_state, target=cfg["target"])

    return {
        "score":        score,
        "total_reward":  round(total_reward, 4),
        "task":          task_name,
        "metrics":       final_state,
    }

