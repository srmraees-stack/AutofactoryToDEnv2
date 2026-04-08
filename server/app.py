"""
app.py — FastAPI server for AutoFactoryToDEnv.

Endpoints
---------
  POST /reset       → ResetResponse
  POST /step        → StepResponse
  GET  /state       → StateResponse
  GET  /score       → ScoreResponse
  GET  /tasks       → TasksListResponse
  POST /evaluate    → EvaluateResponse
"""

from contextlib import asynccontextmanager
import os
import sys

import uvicorn
from fastapi import FastAPI, HTTPException

# Adjust the import path when running via uvicorn from the project root:
#   uvicorn server.app:app --reload
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from models import (
    StepAction, Observation, StepResponse, ResetResponse,
    StateResponse, ScoreResponse,
    TaskConfig, TasksListResponse,
    EvaluateRequest, EvaluateResponse,
)
from server.environment import (
    AutoFactoryToDEnv, compute_score, TASK_CONFIGS, evaluate_policy,
)


# ---------------------------------------------------------------------------
# Shared environment instance (one episode at a time)
# ---------------------------------------------------------------------------

env = AutoFactoryToDEnv()


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Auto-reset once on startup so the env is ready immediately."""
    env.reset()
    yield

app = FastAPI(
    title="AutoFactoryToDEnv",
    description="RL environment server for factory production scheduling.",
    version="2.0.0",
    lifespan=lifespan,
)

@app.get("/health", summary="Health check endpoint for deployments")
def health() -> dict:
    """Returns status ok to satisfy liveness probes."""
    return {"status": "ok"}


# ---------------------------------------------------------------------------
# Core RL routes
# ---------------------------------------------------------------------------

@app.post("/reset", response_model=ResetResponse, summary="Reset the episode")
def reset() -> ResetResponse:
    """Reset the environment and return the initial observation."""
    obs_dict, info = env.reset()
    return ResetResponse(
        observation=Observation(**obs_dict),
        info=info,
    )


@app.post("/step", response_model=StepResponse, summary="Advance one timestep")
def step(action: StepAction) -> StepResponse:
    """
    Apply an action and advance the environment by one hour.

    Raises 400 if the episode is already over.
    """
    try:
        obs_dict, reward, terminated, truncated, info = env.step(
            stamping   = action.stamping,
            molding    = action.molding,
            cnc        = action.cnc,
            compressor = action.compressor,
            welder     = action.welder,
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    obs_dict["hour"] = obs_dict["hour"] % 24

    return StepResponse(
        observation=Observation(**obs_dict),
        reward=reward,
        terminated=terminated,
        truncated=truncated,
        info=info,
    )


# ---------------------------------------------------------------------------
# OpenEnv-spec routes
# ---------------------------------------------------------------------------

@app.get("/state", response_model=StateResponse, summary="Full internal state")
def get_state() -> StateResponse:
    """Return the full internal state (privileged — includes cost, CO₂, etc.)."""
    return StateResponse(**env.state())


@app.get("/score", response_model=ScoreResponse, summary="Deterministic score [0,1]")
def get_score() -> ScoreResponse:
    """Compute the deterministic evaluation score from the current state."""
    return ScoreResponse(score=compute_score(env.state()))


@app.get("/tasks", response_model=TasksListResponse, summary="List task configs")
def list_tasks() -> TasksListResponse:
    """Return all available task difficulty configs (easy, medium, hard)."""
    tasks = [
        TaskConfig(name=name, **cfg) for name, cfg in TASK_CONFIGS.items()
    ]
    return TasksListResponse(tasks=tasks)


@app.post("/evaluate", response_model=EvaluateResponse, summary="Evaluate a greedy policy")
def evaluate(req: EvaluateRequest) -> EvaluateResponse:
    """
    Run a full episode under the requested task using a default greedy policy
    and return the deterministic score + metrics.

    To evaluate custom policies, use the Python ``evaluate_policy()`` function
    directly or implement a custom ``/evaluate`` handler.
    """
    # Default greedy policy: all machines at full power
    def greedy_policy(obs):
        return (2, 2, 2, 1, 1)

    try:
        result = evaluate_policy(greedy_policy, task_name=req.task_name)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    return EvaluateResponse(**result)


def main() -> None:
    """Run the API server entrypoint used by OpenEnv validation."""
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("server.app:app", host="0.0.0.0", port=port)


if __name__ == "__main__":
    main()
