import numpy as np
from train_ppo import AutoFactoryGymEnv
from stable_baselines3 import PPO

# 1. Train model quickly
env = AutoFactoryGymEnv()
model = PPO("MlpPolicy", env, n_steps=2400, batch_size=240, verbose=0, seed=42, device="cpu")
model.learn(total_timesteps=48000)

# 2. Run one deterministic episode with trained model
eval_env = AutoFactoryGymEnv()
obs, _ = eval_env.reset()

print('| Hour | Stmp | Mold | CNC | Comp | Weld | Prod (delta) | Cum Prod | Reward | OvrCap |')
print('|------|------|------|-----|------|------|--------------|----------|--------|--------|')

done = False
while not done:
    action, _ = model.predict(obs, deterministic=True)
    raw_obs_dict = eval_env.env.state()
    hour = raw_obs_dict['hour']
    
    obs, reward, terminated, truncated, info = eval_env.step(action)
    done = terminated or truncated
    
    prod = info['production_delta']
    cum_prod = eval_env.env.production_so_far
    ovrcap = eval_env.env.hours_over_cap
    
    st, mo, cn, co, we = action
    print(f'|  {hour:02d}  |   {st}  |   {mo}  |  {cn}  |   {co}  |   {we}  |       {prod:6.2f} |  {cum_prod:7.2f} | {reward: 6.4f} |     {ovrcap:2d} |')
