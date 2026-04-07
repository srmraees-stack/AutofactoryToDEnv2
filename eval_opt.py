import random
random.seed(42)

from server.environment import AutoFactoryToDEnv, is_peak_hour

env = AutoFactoryToDEnv()
env.reset()

print('| Hour | Policy | Prod (delta) | Cum Prod | OvrCap | Reward |')
print('|------|--------|--------------|----------|--------|--------|')

for h in range(24):
    if is_peak_hour(h):
        action = (1, 1, 1, 0, 1) # Low power on Peak
        pol_str = 'Peak↓'
    else:
        # We simulate a "slower" policy that doesn't sprint to target
        action = (1, 1, 1, 0, 1)
        pol_str = 'Low  '
    
    obs, reward, done, _, info = env.step(*action)
    print(f'| {h:02d}   | {pol_str}  | {info["production_delta"]:>12.2f} | {obs["production_so_far"]:>8.2f} | {getattr(env, "hours_over_cap", 0):2d}     | {reward:>6.4f} |')
