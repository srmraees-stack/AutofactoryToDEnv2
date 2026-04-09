"""
report.py — RL Agent Report for AutoFactoryToDEnv.
-----
  python3 report.py
"""

import sys, os
from server.environment import (
    AutoFactoryToDEnv, compute_final_score, is_peak_hour, get_tariff,
    TARIFF_NIGHT, TARIFF_NORMAL, TARIFF_PEAK,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def tariff_band(hour: int) -> str:
    t = get_tariff(hour)
    if t == TARIFF_NIGHT:  return "NIGHT"
    if t == TARIFF_PEAK:   return "PEAK"
    return "NORMAL"


def machine_label(level: int, machine: str) -> str:
    labels = {
        "stamping":   ["Idle", "Half", "Full"],
        "molding":    ["Idle", "Half", "Full"],
        "cnc":        ["Idle", "Half", "Full"],
        "compressor": ["OFF",  "ON"],
        "welder":     ["OFF",  "Full", "Maint"],
    }
    return labels[machine][level]


def interpret_hour(
    hour: int,
    production_delta: float,
    production_so_far: float,
    action: list,
    target: int = 8_000,
) -> str:
    """Return a one-line decision interpretation for each simulated hour."""
    band = tariff_band(hour)
    all_machines_high = all(a >= 1 for a in action[:3])   # stamping/molding/cnc

    if production_so_far >= target:
        return "✅ Target reached — machines idling to save cost & emissions"

    if production_delta == 0:
        return "⚠️  Idle: No production this hour"

    if band == "NIGHT" and production_delta > 0:
        return "✅ Smart: Using cheap hours for production"

    if band == "PEAK" and all_machines_high:
        return "⚠️  Inefficient: High usage during expensive peak hours"

    return "ℹ️  Balanced operation"


# ---------------------------------------------------------------------------
# Report generator
# ---------------------------------------------------------------------------

def run_report(policy=None, task="medium"):
    """
    Run one full episode and print the hackathon-compliant report.

    Parameters
    ----------
    policy : callable(obs_dict) -> list[int], optional
        Agent policy.  Defaults to a smart heuristic that avoids peak hours.
    task : str
        The task difficulty level ('easy', 'medium', 'hard').
    """
    env = AutoFactoryToDEnv(task=task, enable_breakdowns=(task != "easy"))
    obs, _ = env.reset()
    target = env.target_production

    # Default smart heuristic policy
    def smart_policy(o):
        h = o["hour"]
        # TASK 3: use dynamic target production
        current_target = o.get("production_target", target)
        if o["production_so_far"] >= current_target:
            return [0, 0, 0, 0, 0]           # idle after cap
        if is_peak_hour(h):
            return [1, 1, 1, 0, 1]           # half-power during peak
        return [2, 2, 2, 1, 1]              # full power off-peak

    if policy is None:
        policy = smart_policy

    DIVIDER = "=" * 60

    print()
    print(DIVIDER)
    print("AUTO FACTORY RL AGENT REPORT")
    print("=" * 28)
    print()

    cumulative_cost_inr = 0.0
    all_rewards = []

    for step in range(24):
        hour = obs["hour"]
        band = tariff_band(hour)
        # Use environment's tariff logic (tariff is in INR/MWh)
        tariff_inr = env._get_tariff(hour)

        action = policy(obs)
        obs, reward, terminated, truncated, info = env.step(*action)
        all_rewards.append(reward)

        # Reflect the ACTUAL action (after task overrides)
        actual_action = info.get("actual_action", action)
        prod_delta   = info["production_delta"]
        # Use cost directly from environment (already INR)
        cost_inr     = info["cost_inr"]
        cumulative_cost_inr += cost_inr
        prod_so_far  = obs["production_so_far"]
        current_target = obs.get("production_target", target)

        machine_names = ["stamping", "molding", "cnc", "compressor", "welder"]
        machine_labels = [
            machine_label(actual_action[i], machine_names[i])
            for i in range(5)
        ]

        decision = interpret_hour(hour, prod_delta, prod_so_far, actual_action, target=current_target)

        # ----- per-hour block -----
        print(f"HOUR {hour:02d} | {band} | ₹{tariff_inr:.2f}/kWh")
        print(f"Machines: {machine_labels}")
        print(f"Production: {prod_so_far:.2f} (+{prod_delta:.2f})")
        print(f"Cost: ₹{cumulative_cost_inr:.2f} (+{cost_inr:.2f})")
        print(f"Reward: {reward:.3f}")
        print()
        print(f"## {decision}")
        print()

        if terminated or truncated:
            break

    # ----- final summary -----
    st = env.state()
    final_score = compute_final_score(st, target=st["production_target"])
    target_met  = st["total_production"] >= st["production_target"]

    print(DIVIDER)
    print("📊 FINAL SUMMARY")
    print("=" * 16)
    print()
    print(f"Final Score:       {final_score:.2f}")
    print(f"Final Production:  {st['total_production']:.2f} / {st['production_target']:.2f}")
    print(f"Final Cost:        ₹{cumulative_cost_inr:,.2f}")
    print(f"Final CO₂:         {st['total_co2']:.2f} kg")
    print(f"Avg Machine Health:{sum(st['machine_health'])/len(st['machine_health']):.2%}")
    print(f"✅ Target Achieved: {'YES' if target_met else 'NO'}")
    print("=" * 22)
    print()

    return final_score


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Check if a task mode was passed via CLI
    task_mode = "medium"
    if len(sys.argv) > 1:
        task_mode = sys.argv[1]
    run_report(task=task_mode)
