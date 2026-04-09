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

from fastapi.responses import HTMLResponse

app = FastAPI(
    title="AutoFactoryToDEnv",
    description="RL environment server for factory production scheduling.",
    version="2.0.0",
    lifespan=lifespan,
)

@app.get("/", response_class=HTMLResponse, summary="Landing page")
def root():
    """Welcome page with links to documentation."""
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>AutoFactory API | Home</title>
        <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600&display=swap" rel="stylesheet">
        <style>
            :root {
                --primary: #6366f1;
                --primary-dark: #4f46e5;
                --bg: #0f172a;
                --card-bg: rgba(30, 41, 59, 0.7);
                --text: #f8fafc;
                --text-muted: #94a3b8;
            }
            body {
                margin: 0;
                font-family: 'Outfit', sans-serif;
                background: radial-gradient(circle at top right, #1e293b, #0f172a);
                color: var(--text);
                display: flex;
                align-items: center;
                justify-content: center;
                min-height: 100vh;
                overflow: hidden;
            }
            .container {
                text-align: center;
                background: var(--card-bg);
                backdrop-filter: blur(12px);
                padding: 3rem;
                border-radius: 24px;
                border: 1px solid rgba(255, 255, 255, 0.1);
                box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.5);
                max-width: 500px;
                animation: fadeIn 0.8s ease-out;
            }
            @keyframes fadeIn {
                from { opacity: 0; transform: translateY(20px); }
                to { opacity: 1; transform: translateY(0); }
            }
            h1 {
                font-size: 2.5rem;
                margin-bottom: 0.5rem;
                background: linear-gradient(135deg, #818cf8 0%, #c084fc 100%);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
            }
            p {
                color: var(--text-muted);
                margin-bottom: 2rem;
            }
            .links {
                display: grid;
                gap: 1rem;
            }
            .btn {
                display: block;
                padding: 1rem;
                background: var(--primary);
                color: white;
                text-decoration: none;
                border-radius: 12px;
                font-weight: 600;
                transition: all 0.3s ease;
                border: 1px solid rgba(255, 255, 255, 0.1);
            }
            .btn:hover {
                background: var(--primary-dark);
                transform: translateY(-2px);
                box-shadow: 0 10px 15px -3px rgba(99, 102, 241, 0.4);
            }
            .btn-secondary {
                background: rgba(255, 255, 255, 0.05);
                color: var(--text);
            }
            .btn-secondary:hover {
                background: rgba(255, 255, 255, 0.1);
            }
            .status {
                margin-top: 2rem;
                font-size: 0.875rem;
                display: flex;
                align-items: center;
                justify-content: center;
                gap: 0.5rem;
            }
            .dot {
                width: 8px;
                height: 8px;
                background: #10b981;
                border-radius: 50%;
                box-shadow: 0 0 10px #10b981;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>AutoFactory API</h1>
            <p>Production Scheduling RL Environment</p>
            <div class="links">
                <a href="/docs" class="btn">Explore API Docs</a>
                <a href="/tasks" class="btn btn-secondary">View Task Configurations</a>
                <a href="/health" class="btn btn-secondary">System Health</a>
            </div>
            <div class="status">
                <div class="dot"></div>
                Server is active and running
            </div>
        </div>
    </body>
    </html>
    """

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
    port = int(os.getenv("PORT", "7860"))
    uvicorn.run("server.app:app", host="0.0.0.0", port=port)


if __name__ == "__main__":
    main()
