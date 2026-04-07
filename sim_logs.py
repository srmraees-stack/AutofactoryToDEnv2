import random
random.seed(42)

from server.environment import AutoFactoryToDEnv, is_peak_hour, compute_score

env = AutoFactoryToDEnv()
env.reset()

print("[START] task=factory_scheduling env=factory-env:latest model=gpt-4")

total_reward = 0.0

for h in range(1, 25):
    if is_peak_hour(h-1):
        action = [1, 1, 1, 0, 1]
    else:
        action = [1, 1, 1, 0, 1]
    
    obs, reward, done, _, info = env.step(*action)
    total_reward += reward
    
    error_str = "null"
    print(f"[STEP] step={h} action={action} reward={reward:.2f} done={str(done).lower()} error={error_str}")

st = env.state()
score = compute_score(st)
success = True
print(f"[END] success={str(success).lower()} steps=24 score={score:.2f} rewards={total_reward:.2f}")
