"""
models.py — Pydantic data models for AutoFactoryToDEnv.

Covers:
  - StepAction   : the agent's action at each timestep
  - Observation  : the environment observation returned to the agent
  - StepResponse : full response from /step
  - ResetResponse: full response from /reset
"""

from pydantic import BaseModel, Field
from typing import List, Optional


# ---------------------------------------------------------------------------
# Action
# ---------------------------------------------------------------------------

class StepAction(BaseModel):
    """
    MultiDiscrete action sent by the agent each hour.
    stamping  : 0=idle, 1=half, 2=full
    molding   : 0=idle, 1=half, 2=full
    cnc       : 0=idle, 1=half, 2=full
    compressor: 0=off,  1=on   (boosts output +15%)
    welder    : 0=off,  1=full, 2=maintenance
    """
    stamping:   int = Field(..., ge=0, le=2, description="Stamping machine mode")
    molding:    int = Field(..., ge=0, le=2, description="Molding machine mode")
    cnc:        int = Field(..., ge=0, le=2, description="CNC machine mode")
    compressor: int = Field(..., ge=0, le=1, description="Compressor on/off")
    welder:     int = Field(..., ge=0, le=2, description="Welder mode")


# ---------------------------------------------------------------------------
# Observation
# ---------------------------------------------------------------------------

class Observation(BaseModel):
    """
    Environment state visible to the agent at the start of each hour.

    hour            : current hour index (0–23)
    production_so_far: cumulative parts produced this episode
    machine_health  : health [0.0–1.0] for [stamping, molding, cnc, compressor, welder]
    electricity_price: current tariff in $/MWh
    """
    hour:             int        = Field(..., ge=0, le=23)
    production_so_far: float     = Field(..., ge=0)
    machine_health:   List[float] = Field(..., min_length=5, max_length=5)
    electricity_price: float     = Field(..., gt=0)


# ---------------------------------------------------------------------------
# API responses
# ---------------------------------------------------------------------------

class StepResponse(BaseModel):
    """Returned by POST /step."""
    observation: Observation
    reward:      float
    terminated:  bool
    truncated:   bool
    info:        dict


class ResetResponse(BaseModel):
    """Returned by POST /reset."""
    observation: Observation
    info:        dict


# ---------------------------------------------------------------------------
# State & Score
# ---------------------------------------------------------------------------

class StateResponse(BaseModel):
    """Returned by GET /state — full internal environment state."""
    hour:              int
    total_production:  float
    machine_health:    List[float]
    total_cost:        float
    total_co2:         float
    total_energy:      float
    peak_energy:       float
    breakdown_events:  list


class ScoreResponse(BaseModel):
    """Returned by GET /score — deterministic evaluation score."""
    score: float = Field(..., ge=0, le=1)


# ---------------------------------------------------------------------------
# Task configs
# ---------------------------------------------------------------------------

class TaskConfig(BaseModel):
    """Description of a single task difficulty level."""
    name:                str
    description:         str
    target:              int
    fixed_tariff:        Optional[float]
    enable_breakdowns:   bool
    rush_order:          Optional[dict]
    forced_maintenance:  Optional[dict]
    seed:                int


class TasksListResponse(BaseModel):
    """Returned by GET /tasks."""
    tasks: List[TaskConfig]


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

class EvaluateRequest(BaseModel):
    """Request body for POST /evaluate."""
    task_name: str = Field("medium", description="easy | medium | hard")


class EvaluateResponse(BaseModel):
    """Returned by POST /evaluate."""
    score:        float
    total_reward: float
    task:         str
    metrics:      dict

