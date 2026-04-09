"""
inference.py — Baseline inference script for AutoFactoryToDEnv.

Runs the LLM agent against all 3 tasks and emits structured logs.

Required env vars:
  API_BASE_URL  — LLM endpoint  (default: https://api.openai.com/v1)
  MODEL_NAME    — model id       (default: gpt-4o-mini)
  HF_TOKEN      — API key        (also checked as OPENAI_API_KEY / API_KEY)
"""

import asyncio
import json
import os
import textwrap
from typing import Any, Dict, List, Optional, Sequence, Tuple

from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME   = os.getenv("MODEL_NAME",   "gpt-4o-mini")
API_KEY      = (
    os.getenv("HF_TOKEN")
    or os.getenv("OPENAI_API_KEY")
    or os.getenv("API_KEY")
    or "dummy-key"
)
MAX_STEPS    = 24
TEMPERATURE  = 0.0
MAX_TOKENS   = 120

# Tasks the validator must see
TASKS = [
    {
        "id":         "task_easy",
        "name":       "Simple Order Lookup",
        "difficulty": "easy",
    },
    {
        "id":         "task_medium",
        "name":       "Multi-Step Production Query",
        "difficulty": "medium",
    },
    {
        "id":         "task_hard",
        "name":       "Complex Supply Chain Resolution",
        "difficulty": "hard",
    },
]

SYSTEM_PROMPT = textwrap.dedent("""
    You are controlling a factory production environment for 24 hourly steps.
    Return exactly one JSON array with 5 integers:
    [stamping, molding, cnc, compressor, welder]

    Valid actions:
    - stamping, molding, cnc: 0=idle, 1=half, 2=full
    - compressor: 0=off, 1=on
    - welder: 0=off, 1=full, 2=maintenance

    Objectives:
    - reach the production target by the end of the day
    - avoid expensive peak tariff hours when possible
    - preserve machine health
    - use welder maintenance when health is low

    Respond with JSON only. Example: [2,2,2,1,1]
""").strip()

# ---------------------------------------------------------------------------
# Logging helpers  — MUST match the required format exactly
# ---------------------------------------------------------------------------

def log_start(task_id: str, model: str) -> None:
    print(f"[START] task={task_id} model={model}", flush=True)

def log_step(task_id: str, step: int, action: Any, reward: float, done: bool, error: Optional[str]) -> None:
    line = f"[STEP] task={task_id} step={step} reward={round(float(reward), 4)} done={done}"
    if error:
        line += f" error={error}"
    print(line, flush=True)

def log_end(task_id: str, score: float, steps: int, success: bool) -> None:
    print(f"[END] task={task_id} score={round(float(score), 4)} steps={steps} success={success}", flush=True)

# ---------------------------------------------------------------------------
# Environment wrapper  (HTTP-based — no Docker dependency)
# ---------------------------------------------------------------------------

class EnvHTTPClient:
    """
    Thin wrapper around the FastAPI server's HTTP endpoints.
    Falls back to the local Python environment if the server is unreachable.
    """

    def __init__(self, base_url: str = "http://localhost:7860"):
        self.base_url = base_url.rstrip("/")
        self._use_local = False
        self._local_env = None

    # ------------------------------------------------------------------
    # Try HTTP first; fall back to local Python env
    # ------------------------------------------------------------------
    def _try_connect(self) -> bool:
        try:
            import requests  # noqa: PLC0415
            r = requests.get(f"{self.base_url}/health", timeout=3)
            return r.status_code == 200
        except Exception:
            return False

    def _init_local(self, task_difficulty: str = "easy") -> None:
        """Fall back: instantiate the environment in-process."""
        try:
            from server.environment import AutoFactoryToDEnv  # noqa: PLC0415
            self._local_env = AutoFactoryToDEnv(
                task=task_difficulty,
                enable_breakdowns=(task_difficulty != "easy"),
                production_noise=(task_difficulty != "easy"),
            )
            self._use_local = True
        except Exception as exc:
            raise RuntimeError(f"Cannot connect to server and local import failed: {exc}") from exc

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def reset(self, task_difficulty: str = "easy") -> Tuple[dict, dict]:
        if not self._use_local and not self._try_connect():
            self._init_local(task_difficulty)

        if self._use_local:
            result = self._local_env.reset()
            if isinstance(result, tuple) and len(result) >= 2:
                obs, info = result[0], result[1]
            else:
                obs, info = _as_dict(result), {}
            return _as_dict(obs), _as_dict(info)

        import requests  # noqa: PLC0415
        r = requests.post(f"{self.base_url}/reset", timeout=10)
        r.raise_for_status()
        data = r.json()
        obs  = data.get("observation", data)
        info = data.get("info", {})
        return _as_dict(obs), _as_dict(info)

    def step(self, action: List[int]) -> Tuple[dict, float, bool, bool, dict]:
        if self._use_local:
            result = self._local_env.step(*action)
            obs, reward, terminated, truncated, info = _unpack_step(result)
            return obs, reward, terminated, truncated, info

        import requests  # noqa: PLC0415
        payload = {
            "stamping":   action[0],
            "molding":    action[1],
            "cnc":        action[2],
            "compressor": action[3],
            "welder":     action[4],
        }
        r = requests.post(f"{self.base_url}/step", json=payload, timeout=10)
        r.raise_for_status()
        data = r.json()
        obs        = _as_dict(data.get("observation", {}))
        reward     = float(data.get("reward", 0.0))
        terminated = bool(data.get("terminated", data.get("done", False)))
        truncated  = bool(data.get("truncated", False))
        info       = _as_dict(data.get("info", {}))
        return obs, reward, terminated, truncated, info

    def get_state(self) -> dict:
        if self._use_local:
            return _as_dict(self._local_env.state())
        try:
            import requests  # noqa: PLC0415
            r = requests.get(f"{self.base_url}/state", timeout=5)
            r.raise_for_status()
            return r.json()
        except Exception:
            return {}

    def grade(self, task_id: str, trajectory: list, final_state: dict) -> float:
        """Call /grade/{task_id} or fall back to local grader."""
        if not self._use_local:
            try:
                import requests  # noqa: PLC0415
                payload = {"trajectory": trajectory, "final_state": final_state}
                r = requests.post(f"{self.base_url}/grade/{task_id}", json=payload, timeout=10)
                r.raise_for_status()
                return float(r.json().get("score", 0.0))
            except Exception:
                pass  # fall through to local grader

        # Local grader fallback
        from server.graders import grade_easy_task, grade_medium_task, grade_hard_task  # noqa: PLC0415
        graders = {
            "task_easy":   grade_easy_task,
            "task_medium": grade_medium_task,
            "task_hard":   grade_hard_task,
        }
        grader = graders.get(task_id)
        if grader is None:
            return 0.0
        return float(grader(trajectory, final_state))


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _as_dict(value: Any) -> dict:
    if isinstance(value, dict):
        return value
    for attr in ("model_dump", "dict"):
        fn = getattr(value, attr, None)
        if callable(fn):
            result = fn()
            if isinstance(result, dict):
                return result
    return {}


def _unpack_step(result: Any) -> Tuple[dict, float, bool, bool, dict]:
    if isinstance(result, tuple) and len(result) >= 5:
        obs, reward, terminated, truncated, info = result[:5]
        return _as_dict(obs), float(reward), bool(terminated), bool(truncated), _as_dict(info)
    obs        = _as_dict(getattr(result, "observation", {}))
    reward     = float(getattr(result, "reward", 0.0))
    terminated = bool(getattr(result, "terminated", getattr(result, "done", False)))
    truncated  = bool(getattr(result, "truncated", False))
    info       = _as_dict(getattr(result, "info", {}))
    return obs, reward, terminated, truncated, info


# ---------------------------------------------------------------------------
# Action helpers
# ---------------------------------------------------------------------------

def _heuristic_action(observation: dict) -> List[int]:
    hour             = int(observation.get("hour", 0))
    price            = float(observation.get("electricity_price", 0.0))
    production_so_far = float(observation.get("production_so_far", 0.0))
    production_target = float(observation.get("production_target", 8000.0))
    health            = list(observation.get("machine_health", [1.0] * 5))
    health           += [1.0] * (5 - len(health))

    if production_so_far >= production_target:
        return [0, 0, 0, 0, 0]

    welder_low    = health[4] < 0.45
    any_core_low  = min(health[:3]) < 0.35
    late_day      = hour >= 18
    peak_hour     = price >= 9.85 - 1e-6

    if welder_low and not late_day:
        return [1, 1, 1, 0, 2]
    if any_core_low and peak_hour:
        return [1, 1, 1, 0, 0]
    if peak_hour:
        return [1, 1, 1, 0, 1]
    return [2, 2, 2, 1, 1]


def _extract_action(text: str) -> Optional[List[int]]:
    if not text:
        return None
    start = text.find("[")
    end   = text.rfind("]")
    if start == -1 or end < start:
        return None
    try:
        parsed = json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        return None
    if not isinstance(parsed, list) or len(parsed) != 5:
        return None
    try:
        action = [int(v) for v in parsed]
    except (TypeError, ValueError):
        return None
    bounds = [(0, 2), (0, 2), (0, 2), (0, 1), (0, 2)]
    if any(v < lo or v > hi for v, (lo, hi) in zip(action, bounds)):
        return None
    return action


def _get_model_action(
    client: OpenAI,
    step: int,
    observation: dict,
    last_reward: float,
    rewards: Sequence[float],
) -> Tuple[List[int], Optional[str]]:
    fallback = _heuristic_action(observation)

    if not API_KEY or API_KEY == "dummy-key":
        return fallback, "missing_api_key"

    prompt = (
        f"Step: {step}\n"
        f"Observation: {json.dumps(observation, separators=(',', ':'))}\n"
        f"Last reward: {last_reward:.2f}\n"
        f"Recent rewards: {','.join(f'{r:.2f}' for r in rewards[-5:]) or 'none'}\n"
        "Choose the next 5-integer action array."
    )
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": prompt},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
        )
        raw = completion.choices[0].message.content or ""
        action = _extract_action(raw)
        if action is None:
            return fallback, "invalid_model_response"
        return action, None
    except Exception as exc:
        return fallback, f"llm_error:{type(exc).__name__}"


# ---------------------------------------------------------------------------
# Per-task episode runner
# ---------------------------------------------------------------------------

def run_task(task: dict, env: EnvHTTPClient, client: OpenAI) -> float:
    """
    Run one full episode for the given task.
    Emits [START] … [STEP] … [END] logs.
    Returns final score in [0.0, 1.0].
    """
    task_id    = task["id"]
    difficulty = task["difficulty"]

    log_start(task_id=task_id, model=MODEL_NAME)

    trajectory: List[dict] = []
    rewards:    List[float] = []
    steps_taken = 0
    last_reward = 0.0
    last_error: Optional[str] = None
    score = 0.0
    success = False

    try:
        observation, _ = env.reset(task_difficulty=difficulty)

        for step in range(1, MAX_STEPS + 1):
            action, model_error = _get_model_action(
                client=client,
                step=step,
                observation=observation,
                last_reward=last_reward,
                rewards=rewards,
            )

            obs_next, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            trajectory.append({
                "step":        step,
                "observation": observation,
                "action":      action,
                "reward":      reward,
                "done":        done,
            })
            rewards.append(reward)
            steps_taken = step
            last_reward = reward
            last_error  = (
                info.get("last_action_error") if isinstance(info, dict) else None
            ) or model_error

            log_step(
                task_id=task_id,
                step=step,
                action=action,
                reward=reward,
                done=done,
                error=last_error,
            )

            observation = obs_next
            if done:
                break

        final_state = env.get_state()
        score   = env.grade(task_id, trajectory, final_state)
        score   = min(max(float(score), 0.0), 1.0)
        success = score > 0.0

    except Exception as exc:
        last_error = str(exc)
        success    = False
        score      = 0.0

    log_end(task_id=task_id, score=score, steps=steps_taken, success=success)
    return score


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    server_url = os.getenv("ENV_SERVER_URL", "http://localhost:7860")
    env        = EnvHTTPClient(base_url=server_url)

    all_scores: Dict[str, float] = {}
    for task in TASKS:
        score = run_task(task=task, env=env, client=client)
        all_scores[task["id"]] = score

    # Final summary (informational — not part of the required log format)
    print(
        json.dumps({"type": "SUMMARY", "scores": all_scores}),
        flush=True,
    )


if __name__ == "__main__":
    main()