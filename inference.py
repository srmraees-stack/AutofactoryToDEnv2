import asyncio
import io
import json
import os
import textwrap
from contextlib import redirect_stdout
from typing import Any, List, Optional, Sequence, Tuple

from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables from .env file if it exists
load_dotenv()


try:
    from openenv import MyEnv, compute_score
except ImportError:
    from server.environment import AutoFactoryToDEnv, compute_score as _compute_score

    class MyEnv:
        @classmethod
        async def from_docker_image(cls, image_name: str):
            task_name = os.getenv("TASK_NAME", "medium").lower()
            enable_breakdowns = task_name != "easy"
            production_noise = task_name != "easy"
            return cls(AutoFactoryToDEnv(
                task=task_name,
                enable_breakdowns=enable_breakdowns,
                production_noise=production_noise,
            ))

        def __init__(self, env):
            self._env = env

        async def reset(self):
            return self._env.reset()

        async def step(self, action):
            return self._env.step(*action)

        async def state(self):
            return self._env.state()

        async def close(self):
            close_fn = getattr(self._env, "close", None)
            if callable(close_fn):
                close_fn()

    def compute_score(state):
        return _compute_score(state)


API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
LOCAL_IMAGE_NAME = os.getenv("IMAGE_NAME", "auto-factory-env:latest")
TASK_NAME = os.getenv("TASK_NAME", "medium")
BENCHMARK = os.getenv("BENCHMARK", "autofactory_tod")
MAX_STEPS = 24
TEMPERATURE = 0.0
MAX_TOKENS = 120

SYSTEM_PROMPT = textwrap.dedent(
    """
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
    """
).strip()


def _sanitize_text(value: Any) -> str:
    if value is None:
        return "null"
    return str(value).replace("\r", " ").replace("\n", " ").strip() or "null"


def _suppress_stdout():
    return redirect_stdout(io.StringIO())


def _coerce_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _as_dict(value: Any) -> dict:
    if isinstance(value, dict):
        return value
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        dumped = model_dump()
        return dumped if isinstance(dumped, dict) else {}
    as_dict = getattr(value, "dict", None)
    if callable(as_dict):
        dumped = as_dict()
        return dumped if isinstance(dumped, dict) else {}
    return {}


def _normalize_task_name(task_name: str) -> str:
    task = (task_name or "medium").strip().lower()
    return task if task in {"easy", "medium", "hard"} else "medium"


def _heuristic_action(observation: dict) -> List[int]:
    hour = int(observation.get("hour", 0))
    price = _coerce_float(observation.get("electricity_price"), 0.0)
    production_so_far = _coerce_float(observation.get("production_so_far"), 0.0)
    production_target = _coerce_float(observation.get("production_target"), 8000.0)
    health = list(observation.get("machine_health", [1.0] * 5))
    health += [1.0] * (5 - len(health))

    if production_so_far >= production_target:
        return [0, 0, 0, 0, 0]

    welder_low = health[4] < 0.45
    compressor_low = health[3] < 0.35
    any_core_low = min(health[:3]) < 0.35
    late_day = hour >= 18
    peak_hour = price >= 9.85 - 1e-6

    if welder_low and not late_day:
        return [1, 1, 1, 0, 2]
    if any_core_low and peak_hour:
        return [1, 1, 1, 0, 0]
    if compressor_low:
        return [2, 2, 2, 0, 1]
    if peak_hour:
        return [1, 1, 1, 0, 1]
    return [2, 2, 2, 1, 1]


def _build_user_prompt(
    step: int,
    observation: dict,
    last_reward: float,
    rewards: Sequence[float],
    last_error: Optional[str],
) -> str:
    recent_rewards = ",".join(f"{reward:.2f}" for reward in rewards[-5:]) if rewards else "none"
    health = observation.get("machine_health", [])
    return textwrap.dedent(
        f"""
        Step: {step}
        Observation:
        {json.dumps(observation, separators=(",", ":"), sort_keys=True)}

        Recent rewards: {recent_rewards}
        Last reward: {last_reward:.2f}
        Last error: {_sanitize_text(last_error)}
        Machine health summary: {health}

        Choose the next 5-integer action array.
        """
    ).strip()


def _extract_action_array(raw_text: str) -> Optional[List[int]]:
    if not raw_text:
        return None
    text = raw_text.strip()
    start = text.find("[")
    end = text.rfind("]")
    if start == -1 or end == -1 or end < start:
        return None
    try:
        parsed = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None
    if not isinstance(parsed, list) or len(parsed) != 5:
        return None
    try:
        action = [int(value) for value in parsed]
    except (TypeError, ValueError):
        return None
    bounds = [(0, 2), (0, 2), (0, 2), (0, 1), (0, 2)]
    for value, (lower, upper) in zip(action, bounds):
        if value < lower or value > upper:
            return None
    return action


def _get_model_action(
    client: OpenAI,
    step: int,
    observation: dict,
    last_reward: float,
    rewards: Sequence[float],
    last_error: Optional[str],
) -> Tuple[List[int], Optional[str]]:
    fallback_action = _heuristic_action(observation)
    if not API_KEY:
        return fallback_action, "missing_api_key"

    prompt = _build_user_prompt(
        step=step,
        observation=observation,
        last_reward=last_reward,
        rewards=rewards,
        last_error=last_error,
    )

    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
        )
        content = completion.choices[0].message.content or ""
        action = _extract_action_array(content)
        if action is None:
            return fallback_action, "invalid_model_action"
        return action, None
    except Exception as exc:
        return fallback_action, f"llm_error:{type(exc).__name__}"


def _normalize_reset_result(result: Any) -> Tuple[dict, Any]:
    if isinstance(result, tuple) and len(result) >= 2:
        return _as_dict(result[0]), _as_dict(result[1])
    observation = _as_dict(getattr(result, "observation", {}))
    info = _as_dict(getattr(result, "info", {}))
    return observation, info


def _normalize_step_result(result: Any) -> Tuple[dict, float, bool, bool, dict]:
    if isinstance(result, tuple) and len(result) >= 5:
        observation, reward, terminated, truncated, info = result[:5]
        return _as_dict(observation), _coerce_float(reward), bool(terminated), bool(truncated), _as_dict(info)

    observation = _as_dict(getattr(result, "observation", {}))
    reward = _coerce_float(getattr(result, "reward", 0.0))
    terminated = bool(getattr(result, "terminated", getattr(result, "done", False)))
    truncated = bool(getattr(result, "truncated", False))
    info = _as_dict(getattr(result, "info", {}))
    return observation, reward, terminated, truncated, info


async def _safe_state(env: Any) -> dict:
    state_fn = getattr(env, "state", None)
    if not callable(state_fn):
        return {}
    with _suppress_stdout():
        state = state_fn()
        if asyncio.iscoroutine(state):
            state = await state
    return _as_dict(state)


async def _safe_close(env: Any) -> None:
    close_fn = getattr(env, "close", None)
    if not callable(close_fn):
        return
    with _suppress_stdout():
        result = close_fn()
        if asyncio.iscoroutine(result):
            await result


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={_sanitize_text(task)} env={_sanitize_text(env)} model={_sanitize_text(model)}", flush=True)


def log_step(step: int, action: Sequence[int], reward: float, done: bool, error: Optional[str]) -> None:
    action_str = json.dumps(list(action), separators=(",", ":"))
    done_str = str(bool(done)).lower()
    print(
        f"[STEP] step={step} action={action_str} reward={reward:.2f} done={done_str} error={_sanitize_text(error)}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: Sequence[float]) -> None:
    rewards_str = ",".join(f"{reward:.2f}" for reward in rewards)
    print(
        f"[END] success={str(bool(success)).lower()} steps={steps} score={score:.2f} rewards={rewards_str}",
        flush=True,
    )


async def main() -> None:
    task_name = _normalize_task_name(TASK_NAME)
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY or "missing")

    env = None
    rewards: List[float] = []
    steps_taken = 0
    success = False
    score = 0.0
    last_reward = 0.0
    last_error: Optional[str] = None

    log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)

    try:
        if not LOCAL_IMAGE_NAME:
            raise RuntimeError("missing_local_image_name")

        with _suppress_stdout():
            env = await MyEnv.from_docker_image(LOCAL_IMAGE_NAME)

        with _suppress_stdout():
            reset_result = await env.reset()
        observation, _ = _normalize_reset_result(reset_result)

        for step in range(1, MAX_STEPS + 1):
            action, model_error = _get_model_action(
                client=client,
                step=step,
                observation=observation,
                last_reward=last_reward,
                rewards=rewards,
                last_error=last_error,
            )

            with _suppress_stdout():
                step_result = await env.step(action)

            observation, reward, terminated, truncated, info = _normalize_step_result(step_result)
            done = terminated or truncated

            rewards.append(reward)
            steps_taken = step
            last_reward = reward

            raw_env_error = None
            if isinstance(info, dict):
                raw_env_error = info.get("last_action_error")
            last_error = raw_env_error or model_error

            log_step(
                step=step,
                action=action,
                reward=reward,
                done=done,
                error=last_error,
            )

            if done:
                break

        state = await _safe_state(env) if env is not None else {}
        if state:
            score = _coerce_float(compute_score(state))
        score = min(max(score, 0.0), 1.0)
        success = steps_taken > 0 and last_error is None and 0.0 <= score <= 1.0

    except Exception as exc:
        last_error = _sanitize_text(exc)
        if steps_taken == 0:
            score = 0.0
            success = False

    finally:
        if env is not None:
            try:
                await _safe_close(env)
            except Exception:
                pass
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)


if __name__ == "__main__":
    asyncio.run(main())