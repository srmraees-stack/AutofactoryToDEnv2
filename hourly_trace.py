import sys, os
from server.environment import AutoFactoryToDEnv, is_peak_hour, compute_score

USD_TO_INR = 95.0  # Conversion rate: 1 USD = 95 INR

def run_hourly_trace():
    env = AutoFactoryToDEnv()
    obs, info = env.reset()
    
    total_reward = 0.0
    total_cost_inr = 0.0
    
    print("\n" + "="*110)
    print(f"{'Hour':<6} | {'Tariff (₹)':<12} | {'Production':<20} | {'Cost (₹)':<12} | {'CO2 (kg)':<10} | {'Health % (S,M,C,P,W)':<25} | {'Status'}")
    print("-" * 110)

    for h in range(1, 25):
        # Operational policy: 
        # - Full power during normal/night hours
        # - Half power during peak hours to save cost
        if is_peak_hour(h-1):
            action = [1, 1, 1, 0, 1]  # Half power for production, off for compressor
        else:
            action = [2, 2, 2, 1, 1]  # Full power + compressor
            
        obs, reward, terminated, truncated, info = env.step(*action)
        total_reward += reward
        
        cost_inr = info["cost_usd"] * USD_TO_INR
        total_cost_inr += cost_inr
        tariff_inr = obs["electricity_price"] * USD_TO_INR
        
        health_str = ", ".join([f"{h*100:0.0f}" for h in obs["machine_health"]])
        
        status = "OK"
        if info["breakdown_events"]:
            status = "BREAKDOWN: " + ", ".join(info["breakdown_events"].keys())
        
        print(f"{h-1:02d}:00  | "
              f"{tariff_inr:<12.2f} | "
              f"{info['production_delta']:>7.0f} / {obs['production_so_far']:>7.0f} | "
              f"{cost_inr:<12.2f} | "
              f"{info['co2_kg']:<10.2f} | "
              f"[{health_str:<18}] | "
              f"{status}")

    st = env.state()
    score = compute_score(st)
    
    print("-" * 110)
    print(f"SUMMARY:  Score: {score:.2f}  |  Total Production: {st['total_production']:.0f}  |  Total Cost: ₹{total_cost_inr:,.2f}  |  Total Reward: {total_reward:.2f}")
    print("="*110 + "\n")

if __name__ == "__main__":
    run_hourly_trace()
