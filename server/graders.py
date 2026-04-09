# server/graders.py
"""
Grader functions for AutoFactory TOD Environment.

Each grader signature:
    grade_xxx(trajectory: list[dict], final_state: dict) -> float

- trajectory : list of step dicts (may be empty during validator probing)
- final_state: env.state() dict at episode end (may be empty)
- return      : float in [0.0, 1.0]  — ALWAYS, even on empty input
"""

from __future__ import annotations

from typing import Any


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return bool(value)
    if isinstance(value, str):
        return value.lower() in ("true", "1", "yes")
    return default


# ---------------------------------------------------------------------------
# Easy task grader
# ---------------------------------------------------------------------------

def grade_easy_task(
    trajectory: list[dict[str, Any]],
    final_state: dict[str, Any],
) -> float:
    """
    Easy task: Simple Order Lookup.
    Agent must retrieve a single order's status from the factory system.

    Scoring:
      +0.50  task completed
      +0.25  correct order id identified
      +0.25  correct status returned
      +0.10  bonus for finishing in ≤3 steps
      −0.10  penalty for >10 steps
    """
    # Validator may call with empty inputs — return a valid baseline
    if not trajectory and not final_state:
        return 0.0

    score = 0.0
    num_steps = max(len(trajectory), 1)

    # ---------- derive from final_state ----------
    task_completed  = _safe_bool(final_state.get("task_completed"),  False)
    order_correct   = _safe_bool(final_state.get("order_id_correct"), False)
    status_correct  = _safe_bool(final_state.get("status_correct"),  False)

    # ---------- fallback: infer from trajectory ----------
    if not task_completed and trajectory:
        last = trajectory[-1]
        info = last.get("info", {}) or {}
        task_completed = _safe_bool(info.get("task_completed"), False)

    # ---------- score components ----------
    if task_completed:
        score += 0.50
    if order_correct:
        score += 0.25
    if status_correct:
        score += 0.25

    # Efficiency modifier
    if task_completed and num_steps <= 3:
        score = min(1.0, score + 0.10)
    elif num_steps > 10:
        score = max(0.0, score - 0.10)

    # If nothing else fired, give partial credit based on reward signal
    if score == 0.0 and trajectory:
        total_reward = sum(_safe_float(s.get("reward", 0.0)) for s in trajectory)
        if total_reward > 0:
            score = min(0.3, total_reward / (num_steps * 10.0))

    return round(min(1.0, max(0.0, score)), 4)


# ---------------------------------------------------------------------------
# Medium task grader
# ---------------------------------------------------------------------------

def grade_medium_task(
    trajectory: list[dict[str, Any]],
    final_state: dict[str, Any],
) -> float:
    """
    Medium task: Multi-Step Production Query.
    Agent must handle a multi-turn dialogue to resolve a production
    scheduling conflict.

    Scoring:
      +0.20  conflict identified
      +0.20  resolution proposed
      +0.30  resolution correct
      +0.20  task completed
      +0.10  no contradictions in responses
      −0.15  penalty for >15 steps
    """
    if not trajectory and not final_state:
        return 0.0

    score     = 0.0
    num_steps = max(len(trajectory), 1)

    conflict_identified  = _safe_bool(final_state.get("conflict_identified"),  False)
    resolution_proposed  = _safe_bool(final_state.get("resolution_proposed"),  False)
    resolution_correct   = _safe_bool(final_state.get("resolution_correct"),   False)
    task_completed       = _safe_bool(final_state.get("task_completed"),       False)
    no_contradictions    = _safe_bool(final_state.get("no_contradictions"),    True)

    if conflict_identified:
        score += 0.20
    if resolution_proposed:
        score += 0.20
    if resolution_correct:
        score += 0.30
    if task_completed:
        score += 0.20
    if no_contradictions:
        score += 0.10

    if num_steps > 15:
        score = max(0.0, score - 0.15)

    # Reward-based partial credit
    if score == 0.0 and trajectory:
        total_reward = sum(_safe_float(s.get("reward", 0.0)) for s in trajectory)
        if total_reward > 0:
            score = min(0.4, total_reward / (num_steps * 8.0))

    return round(min(1.0, max(0.0, score)), 4)


# ---------------------------------------------------------------------------
# Hard task grader
# ---------------------------------------------------------------------------

def grade_hard_task(
    trajectory: list[dict[str, Any]],
    final_state: dict[str, Any],
) -> float:
    """
    Hard task: Complex Supply Chain Resolution.
    Agent must coordinate inventory, scheduling, and logistics to resolve
    a multi-constraint supply chain disruption.

    Scoring:
      +0.10  inventory checked
      +0.15  schedule updated
      +0.15  logistics notified
      +0.25  all constraints satisfied
      +0.20  task completed
      +0.05  no deadlock
      +0.10  cost-optimal resolution
      −0.20  penalty for >25 steps
      −0.30  penalty for deadlock
    """
    if not trajectory and not final_state:
        return 0.0

    score     = 0.0
    num_steps = max(len(trajectory), 1)

    inventory_checked        = _safe_bool(final_state.get("inventory_checked"),        False)
    schedule_updated         = _safe_bool(final_state.get("schedule_updated"),         False)
    logistics_notified       = _safe_bool(final_state.get("logistics_notified"),       False)
    all_constraints_satisfied = _safe_bool(final_state.get("all_constraints_satisfied"), False)
    task_completed           = _safe_bool(final_state.get("task_completed"),           False)
    no_deadlock              = _safe_bool(final_state.get("no_deadlock"),              True)
    cost_optimal             = _safe_bool(final_state.get("cost_optimal"),             False)

    if inventory_checked:
        score += 0.10
    if schedule_updated:
        score += 0.15
    if logistics_notified:
        score += 0.15
    if all_constraints_satisfied:
        score += 0.25
    if task_completed:
        score += 0.20
    if no_deadlock:
        score += 0.05
    if cost_optimal:
        score += 0.10

    if num_steps > 25:
        score = max(0.0, score - 0.20)
    if not no_deadlock:
        score = max(0.0, score - 0.30)

    # Reward-based partial credit
    if score == 0.0 and trajectory:
        total_reward = sum(_safe_float(s.get("reward", 0.0)) for s in trajectory)
        if total_reward > 0:
            score = min(0.5, total_reward / (num_steps * 6.0))

    return round(min(1.0, max(0.0, score)), 4)