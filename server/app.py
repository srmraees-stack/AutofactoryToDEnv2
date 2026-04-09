"""
app.py — FastAPI server for AutoFactoryToDEnv.

Endpoints
---------
  POST /reset           → ResetResponse
  POST /step            → StepResponse
  GET  /state           → StateResponse
  GET  /score           → ScoreResponse
  GET  /tasks           → task list
  POST /grade/{task_id} → grader result
  GET  /health          → liveness probe
"""

# ---------------------------------------------------------------------------
# Fix sys.path FIRST — before any local imports
# ---------------------------------------------------------------------------
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ---------------------------------------------------------------------------
# Standard imports
# ---------------------------------------------------------------------------
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse

# ---------------------------------------------------------------------------
# Local imports (after sys.path fix)
# ---------------------------------------------------------------------------
from models import (
    StepAction,
    Observation,
    StepResponse,
    ResetResponse,
    StateResponse,
    ScoreResponse,
)
from server.environment import AutoFactoryToDEnv, compute_score
from server.graders import grade_easy_task, grade_medium_task, grade_hard_task

# ---------------------------------------------------------------------------
# Grader registry
# ---------------------------------------------------------------------------
GRADERS = {
    "task_easy":   grade_easy_task,
    "task_medium": grade_medium_task,
    "task_hard":   grade_hard_task,
}

TASK_METADATA = [
    {
        "id":          "task_easy",
        "name":        "Simple Order Lookup",
        "difficulty":  "easy",
        "description": "Agent must retrieve a single order status from the factory system",
        "grader":      "server.graders:grade_easy_task",
    },
    {
        "id":          "task_medium",
        "name":        "Multi-Step Production Query",
        "difficulty":  "medium",
        "description": "Agent must handle a multi-turn dialogue to resolve a production scheduling conflict",
        "grader":      "server.graders:grade_medium_task",
    },
    {
        "id":          "task_hard",
        "name":        "Complex Supply Chain Resolution",
        "difficulty":  "hard",
        "description": "Agent must coordinate across inventory, scheduling, and logistics to resolve a supply chain disruption",
        "grader":      "server.graders:grade_hard_task",
    },
]

# ---------------------------------------------------------------------------
# Shared environment instance
# ---------------------------------------------------------------------------
env = AutoFactoryToDEnv()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Auto-reset on startup so the env is immediately ready."""
    env.reset()
    yield


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(
    title="AutoFactoryToDEnv",
    description="RL environment server for factory production scheduling.",
    version="2.0.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Landing page
# ---------------------------------------------------------------------------
@app.get("/", response_class=HTMLResponse, summary="Landing page")
def root():
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <title>AutoFactory API</title>
        <style>
            body { font-family: sans-serif; background: #0f172a; color: #f8fafc;
                   display: flex; align-items: center; justify-content: center;
                   min-height: 100vh; margin: 0; }
            .box { text-align: center; padding: 2rem; border-radius: 1rem;
                   background: rgba(30,41,59,.8); max-width: 420px; }
            a { display: block; margin: .5rem 0; padding: .75rem; border-radius: .5rem;
                background: #6366f1; color: #fff; text-decoration: none; }
            a:hover { background: #4f46e5; }
        </style>
    </head>
    <body>
        <div class="box">
            <h1>AutoFactory API</h1>
            <p>Production Scheduling RL Environment</p>
            <a href="/docs">API Docs</a>
            <a href="/tasks">Tasks</a>
            <a href="/health">Health</a>
        </div>
    </body>
    </html>
    """


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------
@app.get("/health", summary="Liveness probe")
def health() -> dict:
    return {"status": "ok"}


# ---------------------------------------------------------------------------
# Core RL routes
# ---------------------------------------------------------------------------
@app.post("/reset", response_model=ResetResponse, summary="Reset the episode")
def reset() -> ResetResponse:
    obs_dict, info = env.reset()
    return ResetResponse(
        observation=Observation(**obs_dict),
        info=info,
    )


@app.post("/step", response_model=StepResponse, summary="Advance one timestep")
def step(action: StepAction) -> StepResponse:
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
    return StateResponse(**env.state())


@app.get("/score", response_model=ScoreResponse, summary="Deterministic score [0,1]")
def get_score() -> ScoreResponse:
    return ScoreResponse(score=compute_score(env.state()))


# ---------------------------------------------------------------------------
# Task & Grader routes  (required by OpenEnv validator)
# ---------------------------------------------------------------------------
@app.get("/tasks", summary="List all tasks with grader info")
def list_tasks() -> dict:
    """Return all 3 tasks. The validator enumerates this endpoint."""
    return {"tasks": TASK_METADATA}


@app.post("/grade/{task_id}", summary="Grade a completed episode")
def grade_task(task_id: str, payload: dict) -> dict:
    """
    Accepts { trajectory: [...], final_state: {...} }
    Returns  { task_id, score, reward }
    """
    if task_id not in GRADERS:
        raise HTTPException(status_code=404, detail=f"Task '{task_id}' not found. Valid: {list(GRADERS)}")

    trajectory  = payload.get("trajectory", [])
    final_state = payload.get("final_state", {})

    score = float(GRADERS[task_id](trajectory, final_state))
    score = min(max(score, 0.0), 1.0)

    return {"task_id": task_id, "score": score, "reward": score}


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------
def main() -> None:
    port = int(os.getenv("PORT", "7860"))
    uvicorn.run("server.app:app", host="0.0.0.0", port=port, reload=False)


if __name__ == "__main__":
    main()