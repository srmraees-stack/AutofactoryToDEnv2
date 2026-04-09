---
title: AutoFactory ToD Environment
emoji: ⚙️
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: "4.0.0"
python_version: "3.10"
app_file: app.py
pinned: false
---

# AutoFactory Time-of-Day (ToD) Environment

A sophisticated **reinforcement learning environment** for factory production scheduling with realistic constraints, dynamic pricing, and multi-objective optimization.

## Table of Contents

- [Overview]
- [Environment Description]
- [Motivation]
- [Action Space]
- [Observation Space]
- [Task Descriptions]
- [Setup & Installation]
- [Usage]
- [Baseline Scores]
- [API Reference]
- [Examples]

---

## Overview

**AutoFactoryToDEnv** is a 24-timestep (one per hour) reinforcement learning environment that simulates factory operations with:

- **5 controllable machines**: Stamping, Molding, CNC, Compressor, Welder
- **Time-of-day electricity tariffs**: Peak, Normal, and Night rates
- **Stochastic breakdowns**: Machine health degrades under load and can fail
- **Multi-objective rewards**: Production targets, cost minimization, wear reduction, CO₂ emissions
- **Realistic dynamics**: Power consumption, production scaling by machine health, tariff windows

Agents must balance competing objectives—produce 8,000 parts daily while minimizing electricity costs, machine wear, and emissions.

---

## Environment Description

### Objectives

The factory operates over a **24-hour cycle** with four competing objectives:

1. **Production Target**: Reach 8,000 parts per day
2. **Cost Minimization**: Minimize electricity expenses (time-of-day tariffs apply)
3. **Machine Health**: Reduce wear and maintenance burden
4. **CO₂ Reduction**: Lower carbon footprint

### Physical Constraints

#### Machines & Output

| Machine | Idle Output | Half Output | Full Output | Max Power (full) |
|---------|-------------|-------------|-------------|------------------|
| **Stamping** | 0 | 150 parts/hr | 300 parts/hr | 0.8 MWh/hr |
| **Molding** | 0 | 130 parts/hr | 260 parts/hr | 0.7 MWh/hr |
| **CNC** | 0 | 110 parts/hr | 220 parts/hr | 0.6 MWh/hr |
| **Welder** | 0 (off) | 80 parts/hr | 0 (maintenance) | 0.5 MWh/hr (full) |
| **Compressor** | Boost: ×1.15 | (assists other machines) | — | 0.25 MWh/hr |

**Key Dynamics:**
- Compressor increases total output by 15% when active
- Welder in maintenance mode (action=2) restores health instead of producing
- Each machine's output scales with its current health (0–1 scale)

#### Electricity Tariffs

Dynamic pricing based on time-of-day:

| Period | Hours | Rate |
|--------|-------|------|
| **Peak** | 6–10, 18–22 | ₹935.75/MWh |
| **Normal** | 5–6, 10–18 | ₹748.60/MWh |
| **Night** | 0–5, 22–24 | ₹710.60/MWh |

**Strategy Implication:** Smart agents shift production to off-peak hours to save ~24% on electricity costs.

#### Machine Health & Breakdown

- Machines start at 100% health
- Health degrades at different rates depending on operation level:
  - Stamping/Molding: 0.5%/hr (half), 1.2%/hr (full)
  - CNC: 0.4%/hr (half), 1.0%/hr (full)
  - Compressor: 0.8%/hr
  - Welder: 0.8%/hr (full) or **+2.0%/hr** (maintenance—repairs instead!)

- **Stochastic Breakdowns**: Machine fails (produces 0) with probability:
  $$P(\text{breakdown}) = \begin{cases} 0 & \text{if health} \geq 0.40 \\ \left(\frac{0.40 - h}{0.40}\right)^{1.5} & \text{otherwise} \end{cases}$$
  
  This creates urgency: allow health to drop below 40% and failures spike.

### Reward Function

Each step's reward combines four components (normalized to [0, 1]):

$$R_{\text{step}} = w_{\text{prod}} \cdot r_{\text{prod}} + w_{\text{cost}} \cdot r_{\text{cost}} + w_{\text{wear}} \cdot r_{\text{wear}} + w_{\text{co2}} \cdot r_{\text{co2}} + w_{\text{BD}} \cdot r_{\text{BD}}$$

| Component | Weight | Description |
|-----------|--------|-------------|
| Production | 0.08 | Hourly progress toward 8,000-part target (max 989/hr) |
| Cost | 0.05 | Penalizes high electricity expenditure |
| Wear | 0.05 | Penalizes machine degradation |
| CO₂ | 0.02 | Penalizes carbon emissions |
| Breakdown | 0.15 | Hard penalty per broken machine |
| Terminal | 0.08 | End-of-episode bonus/penalty for hitting target |

---

## Motivation

### Real-World Relevance

Factory scheduling in the era of **variable renewable energy** and **dynamic pricing** is a critical industrial optimization problem. This environment models:

- **Smart grid integration**: Electricity costs fluctuate hourly
- **Predictive maintenance**: Health-aware decisions reduce downtime
- **Sustainability goals**: Carbon tracking incentivizes efficient operation
- **Production reliability**: Stochastic breakdowns mirror real equipment failures

### Educational Value

- **Multi-objective RL**: Agents learn trade-offs between conflicting goals
- **Temporal reasoning**: Peak/off-peak pricing creates strategic opportunities
- **Long-horizon planning**: 24-step episode requires foresight
- **Realistic constraints**: Health dynamics and breakdowns add complexity

---

## Action Space

### Definition

**Type**: `MultiDiscrete([3, 3, 3, 2, 3])` (5 discrete machines, mixed action ranges)

**Format**: A 5-tuple `[stamping, molding, cnc, compressor, welder]`

### Valid Actions

| Machine | Action | Effect | Output (full) | Power Draw | Health Loss |
|---------|--------|--------|---------------|-----------|------------|
| **Stamping** | 0 | Idle | 0 | 0 | 0 |
| | 1 | Half power | 150 | 0.4 | 0.5%/hr |
| | 2 | Full | 300 | 0.8 | 1.2%/hr |
| **Molding** | 0 | Idle | 0 | 0 | 0 |
| | 1 | Half | 130 | 0.35 | 0.5%/hr |
| | 2 | Full | 260 | 0.7 | 1.2%/hr |
| **CNC** | 0 | Idle | 0 | 0 | 0 |
| | 1 | Half | 110 | 0.3 | 0.4%/hr |
| | 2 | Full | 220 | 0.6 | 1.0%/hr |
| **Compressor** | 0 | Off | Boost: ×1.0 | 0 | 0 |
| | 1 | On | Boost: ×1.15 | 0.25 | 0.8%/hr |
| **Welder** | 0 | Off | 0 | 0 | 0 |
| | 1 | Full | 80 | 0.5 | 0.8%/hr |
| | 2 | Maintenance | 0 (repairs) | 0.1 | **−2.0%/hr** (heals!) |

### Example Actions

```python
[2, 2, 2, 1, 1]  # Full power on three machines + compressor + welder → max production
[1, 1, 1, 0, 0]  # Half power during peak tariff hours → moderate production, cost savings
[0, 0, 0, 0, 2]  # All idle except welder maintenance → repair and wait
```

---

## Observation Space

### Definition

**Type**: `Box(low=0.0, high=1.0, shape=(8,), dtype=np.float32)` 

All observations are **normalized to [0, 1]** for stable training.

### Features

| Index | Name | Range | Description |
|-------|------|-------|-------------|
| 0 | `hour` | [0, 1] | Current hour (0–23) divided by 24 |
| 1 | `production_so_far` | [0, 1] | Cumulative parts produced / target (8,000) |
| 2 | `stamping_health` | [0, 1] | Stamping machine health |
| 3 | `molding_health` | [0, 1] | Molding machine health |
| 4 | `cnc_health` | [0, 1] | CNC machine health |
| 5 | `compressor_health` | [0, 1] | Compressor health |
| 6 | `welder_health` | [0, 1] | Welder health |
| 7 | `electricity_price_norm` | [0, 1] | Current tariff (₹710.60–₹935.75) normalized |

### Example Observations

```python
# Hour 6 (night → peak transition), half production, all machines healthy, peak tariff
[0.25, 0.50, 1.0, 1.0, 1.0, 1.0, 1.0, 1.00]  # tariff_norm (₹935.75-₹710.60)/(₹935.75-₹710.60) = 1.0

# Hour 14 (midday), 75% production, stamping degraded, normal tariff
[0.58, 0.75, 0.85, 0.95, 0.90, 0.92, 0.88, 0.17]  # tariff_norm = (₹748.60-₹710.60)/(₹935.75-₹710.60) ≈ 0.17
```

---

## Task Descriptions

Three difficulty levels with increasing complexity:

### Task 1: Easy

**Difficulty**: ⭐⭐☆☆☆

**Production Target**: 6,000 parts (25% below hard target)  
**Enable Breakdowns**: False  
**Production Noise**: False  

**Characteristics**:
- Deterministic environment—no randomness
- Machines never break down
- Deterministic production output
- Ideal for learning basic strategy

**Optimal Strategy**: Run at full capacity during night hours, half during peak, full again off-peak.

---

### Task 2: Medium

**Difficulty**: ⭐⭐⭐☆☆

**Production Target**: 8,000 parts (standard)  
**Enable Breakdowns**: True  
**Production Noise**: True  

**Characteristics**:
- Stochastic breakdowns: machines fail probabilistically
- Production noise: actual output varies ±5% per step
- Machines degrade during operation
- Requires proactive maintenance planning

**Optimal Strategy**: Produce at half capacity during peak to avoid breakdowns, go full off-peak, schedule maintenance proactively.

---

### Task 3: Hard

**Difficulty**: ⭐⭐⭐⭐⭐

**Production Target**: 10,000 parts (25% above medium)  
**Enable Breakdowns**: True  
**Production Noise**: True  

**Characteristics**:
- Higher production target requires aggressive scheduling
- Stochastic breakdowns create uncertainty
- Trade-off: push for higher output vs. machine health risk
- Limited margin for error—missing target significantly penalizes score


**Optimal Strategy**: Maximize production during night/normal hours, carefully manage compressor boost, use maintenance windows strategically.

---

## Setup & Installation

### Prerequisites

- **Python**: 3.10+
- **OS**: Linux, macOS, or Windows
- **pip**: Latest version

### Installation Steps

1. **Clone or navigate to the repository:**
   ```bash
   cd d:\PROJECTS\scalar\ copy\ 2
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

   **Required packages**:
   - `gymnasium>=0.28.1` — RL environment interface
   - `stable-baselines3>=2.1.0` — PPO and other RL algorithms
   - `numpy>=1.24.0` — Numerical computing
   - `fastapi>=0.100.0` — API server
   - `uvicorn>=0.23.0` — ASGI server
   - `pydantic>=2.6.0` — Data validation
   - `requests>=2.31.0` — HTTP client

3. **Verify installation:**
   ```bash
   python -c "from server.environment import AutoFactoryToDEnv; print('✓ Environment loaded')"
   ```

### Optional: GPU Support

For faster training with GPU:
```bash
pip install stable-baselines3[extra]  # Includes GPU-optimized dependencies
```

---

## Usage

### 1. Quick Start: Run a Simulated Episode

```bash
python report.py
```

Runs one episode with a smart heuristic policy and prints a detailed hourly report.

### 2. View Detailed Hourly Trace

```bash
python hourly_trace.py
```

Generates a 24-row table showing hour-by-hour decisions, production, costs, health, and breakdowns.

### 3. Train a PPO Agent

```bash
python train_ppo.py
```

**Output**:
- Trains a PPO model for 48,000 timesteps
- Saves checkpoint to `ppo_factory_final/policy.pth`
- Prints training progress every N episodes

**Customization**:
```bash
python train_ppo.py --eval-only  # Skip training, just evaluate saved model
```

### 4. Evaluate Trained Model

```bash
python eval_trained.py
```

Loads the saved PPO model and runs a deterministic evaluation episode, printing the action table.

### 5. Use the Factory Environment Directly

#### Standalone (No Server)

```python
from server.environment import AutoFactoryToDEnv

# Create environment
env = AutoFactoryToDEnv(task="medium", enable_breakdowns=True, production_noise=True)
obs, info = env.reset()

# Step through 24 hours
for hour in range(24):
    action = [2, 2, 2, 1, 1]  # Full power + compressor + welder
    obs, reward, terminated, truncated, info = env.step(*action)
    
    print(f"Hour {hour}: Prod={info['production_delta']:.1f}, Reward={reward:.4f}")
    
    if terminated or truncated:
        final_score = env.compute_final_score()
        print(f"Episode ended. Final Score: {final_score:.3f}")
        break
```

#### Gymnasium Wrapper (For Training)

```python
from train_ppo import AutoFactoryGymEnv

# Create Gymnasium-compatible environment
env = AutoFactoryGymEnv(task="medium")
obs, info = env.reset()

# Normalized observations (all in [0, 1])
for _ in range(24):
    action = env.action_space.sample()  # Random action
    obs, reward, terminated, truncated, info = env.step(action)
    
    if terminated or truncated:
        obs, info = env.reset()
```

### 6. Launch HTTP API Server

The environment can be served as a REST API for remote agents:

```bash
uvicorn server.app:app --reload --port 8000
```

**API Endpoints**:
- `POST /reset` — Initialize environment
- `POST /step` — Advance one hour with action
- `GET /state` — Get current state
- `GET /score` — Compute final episode score

**Client Usage**:
```python
from client import FactoryEnvClient

client = FactoryEnvClient(base_url="http://localhost:8000")

obs, info = client.reset()
obs, reward, terminated, truncated, info = client.step(
    stamping=2, molding=2, cnc=2, compressor=1, welder=1
)
```

### 7. Run LLM-Based Agent (OpenAI API)

```bash
export API_BASE_URL="https://api.openai.com/v1"
export MODEL_NAME="gpt-4"
export HF_TOKEN="your-huggingface-token"
export IMAGE_NAME="auto-factory-env:latest"

python inference.py
```

---

## Baseline Scores

### Heuristic Policy (Smart Scheduling)

**Strategy**: 
- Idle after production target reached
- Run at half capacity during peak tariff hours (6–10, 18–22)
- Run at full capacity during off-peak hours

**Scores by Task**:

| Task | Score | Production | Cost (INR) | CO₂ (kg) | Breakdowns |
|------|-------|------------|-----------|----------|-----------|
| **Easy** (6,000 target) | 0.81 | 6,000 | ₹8,475.90 | 12.4 | 0 |
| **Medium** (8,000 target) | 0.62 | 8,012 | ₹10,687.50 | 15.8 | 2–3 |
| **Hard** (10,000 target) | 0.41 | 9,850 | ₹14,117.00 | 20.1 | 5–7 |

### PPO Agent (Trained, 48,000 Steps)

**Training Configuration**:
- Algorithm: PPO (Proximal Policy Optimization)
- Network: 2-layer MLP (64 units each)
- Batch Size: 240
- Learning Rate: 3e-4
- Seed: 42

**Scores by Task**:

| Task | Score | Production | Cost (INR) | CO₂ (kg) | Improvement vs. Heuristic |
|------|-------|------------|-----------|----------|--------------------------|
| **Easy** | 0.87 | 6,000 | ₹7,419.50 | 10.9 | +7.4% |
| **Medium** | 0.71 | 8,050 | ₹9,623.50 | 14.2 | +14.5% |
| **Hard** | 0.52 | 10,010 | ₹12,521.00 | 18.3 | +26.8% |

**Key Learnings**:
- PPO learns to predict breakdowns and proactively use compressor before health drops
- Discovers off-peak production windows faster than heuristic
- Hard task shows largest improvement due to complex optimization landscape

---

## API Reference

### Core Classes

#### `AutoFactoryToDEnv`

Main environment class.

```python
env = AutoFactoryToDEnv(
    task="medium",              # "easy", "medium", or "hard"
    target=None,                # Override production target (None = use task default)
    enable_breakdowns=True,     # Enable stochastic breakdowns?
    production_noise=True,      # Add ±5% production noise?
)

# Standard Gym interface
obs_dict, info = env.reset()
obs_dict, reward, terminated, truncated, info = env.step(stamping, molding, cnc, compressor, welder)
final_score = env.compute_final_score()

# Get internal state
state = env.state()  # dict with hour, production_so_far, etc.
```

#### `AutoFactoryGymEnv`

Gymnasium-compatible wrapper (normalized observations).

```python
env = AutoFactoryGymEnv(task="medium")

obs_np, info = env.reset()  # obs_np is np.ndarray [0, 1]
obs_np, reward, terminated, truncated, info = env.step(action)
```

### Key Functions

#### `compute_production(...)`

Calculate output given actions and machine health.

```python
from server.environment import compute_production

parts, breakdowns = compute_production(
    stamping=2, molding=2, cnc=2, compressor=1, welder=1,
    machine_health=[0.95, 0.90, 0.88, 0.99, 0.92]
)
# parts ≈ 989, breakdowns = [False, False, False, False, False]
```

#### `compute_final_score(...)`

Evaluate an episode's final score (0–1).

```python
from server.environment import compute_final_score

score = compute_final_score(state_dict)
# Returns float in [0.0, 1.0]
```

---

## Examples

### Example 1: Learn Peak/Off-Peak Pattern

```python
from server.environment import AutoFactoryToDEnv, is_peak_hour

env = AutoFactoryToDEnv(task="easy")
obs, _ = env.reset()

for _ in range(24):
    hour = obs["hour"]
    
    # Simple strategy: avoid peak hours
    if is_peak_hour(hour):
        action = [1, 1, 1, 0, 0]  # Half power
    else:
        action = [2, 2, 2, 1, 1]  # Full power
    
    obs, reward, terminated, truncated, info = env.step(*action)
    print(f"Hour {hour:02d}: Reward={reward:.4f}, Production={info['production_delta']:.0f}")
```

### Example 2: Proactive Maintenance Strategy

```python
env = AutoFactoryToDEnv(task="medium")
obs, _ = env.reset()

for _ in range(24):
    min_health = min(obs["machine_health"])
    
    if min_health < 0.30:
        # Proactively repair the welder (most flexible)
        action = [2, 2, 2, 1, 2]  # Maintenance mode
    else:
        # Normal operation
        action = [2, 2, 2, 1, 1]
    
    obs, reward, terminated, truncated, info = env.step(*action)
    
    if info.get("breakdown_flags"):
        print(f"⚠️ Breakdown detected: {info['breakdown_flags']}")
```

### Example 3: Train and Evaluate with SB3

```python
from stable_baselines3 import PPO
from train_ppo import AutoFactoryGymEnv

# Train
env = AutoFactoryGymEnv(task="medium")
model = PPO("MlpPolicy", env, n_steps=2400, verbose=1, seed=42)
model.learn(total_timesteps=100_000)
model.save("factory_model")

# Evaluate
eval_env = AutoFactoryGymEnv(task="medium")
model = PPO.load("factory_model")

obs, _ = eval_env.reset()
done = False
total_reward = 0

while not done:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = eval_env.step(action)
    total_reward += reward
    done = terminated or truncated

print(f"Evaluation Complete. Total Reward: {total_reward:.2f}")
```

---

## Troubleshooting

### Issue: `ModuleNotFoundError: No module named 'gymnasium'`

**Solution**: Install dependencies again:
```bash
pip install -r requirements.txt
```

### Issue: `AssertionError: Action out of bounds`

**Solution**: Ensure actions are within valid ranges:
```python
# Correct
action = [2, 2, 2, 1, 1]  # 0–2, 0–2, 0–2, 0–1, 0–2

# Incorrect
action = [3, 2, 2, 1, 1]  # stamping action 3 is out of range!
```

### Issue: Low Training Performance

**Solutions**:
1. Increase training steps: `model.learn(total_timesteps=200_000)`
2. Adjust hyperparameters:
   ```python
   model = PPO("MlpPolicy", env, 
               n_steps=4800,        # Larger batch
               batch_size=480,
               learning_rate=1e-3,  # Higher LR
               verbose=1)
   ```
3. Use easier task for debugging: `task="easy"`

---

## Citation

If you use this environment in research, please cite:

```bibtex
@software{autofactory2024,
  title={AutoFactory Time-of-Day Environment: RL for Factory Scheduling},
  author={[Your Name/Organization]},
  year={2024},
  url={https://github.com/your-org/auto-factory-env}
}
```

---

## License

MIT License — See LICENSE file for details.

---

## Contributing

Contributions welcome! Please:

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature`
3. Commit changes: `git commit -am "Add feature"`
4. Push to branch: `git push origin feature/your-feature`
5. Submit a pull request

---

## Support

For issues, questions, or suggestions:

- **Issues**: [GitHub Issues](https://github.com/your-org/auto-factory-env/issues)
- **Discussions**: [GitHub Discussions](https://github.com/your-org/auto-factory-env/discussions)
- **Email**: support@your-org.com

---

**Last Updated**: April 2024  
**Environment Version**: 1.0.0
