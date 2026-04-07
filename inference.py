import os
import json
import asyncio
import traceback
from openai import AsyncOpenAI

# Ensure fallback implementation if openenv is not provided in environment
try:
    from openenv import MyEnv, compute_score
except ImportError:
    # Dummy mock implementations to prevent syntax crashing during unexpected execution
    class MyEnv:
        @classmethod
        async def from_docker_image(cls, image_name: str):
            raise NotImplementedError("MyEnv must be provided by the hackathon runner.")
    
    def compute_score(*args, **kwargs):
        return 0.0


PROMPT_TEMPLATE = """
You are controlling a factory to reach an 8000 production target over 24 hours.
You must output an action for the 5 machines: [stamping, molding, cnc, compressor, welder].
Valid discrete actions:
- stamping, molding, cnc: 0(idle), 1(half), 2(full)
- compressor: 0(off), 1(on)
- welder: 0(off), 1(full), 2(maintenance to repair health)

Observation Data:
{obs}

Output ONLY a JSON list of 5 integers. Example: [2, 2, 2, 1, 1]
"""

async def run_inference():
    # 1. Safely read required environment variables
    api_base_url = os.environ["API_BASE_URL"]
    model_name   = os.environ["MODEL_NAME"]
    hf_token     = os.environ["HF_TOKEN"]
    image_name   = os.environ["IMAGE_NAME"]
    
    # Track final metrics
    success = False
    total_steps = 0
    step_rewards = []
    score = 0.0
    
    # Hackathon evaluation metrics mandate strict STDOUT formatting
    print(f"[START] task=factory_scheduling env={image_name} model={model_name}")
    
    try:
        # 2. Instantiate OpenAI Client
        client = AsyncOpenAI(
            base_url=api_base_url,
            api_key=hf_token,  # Common pattern for proxy endpoints using HF token
        )

        # 3. Spin up evaluation environment
        env = await MyEnv.from_docker_image(image_name)
        obs, _ = await env.reset()
        
        done = False
        while not done and total_steps < 24:
            total_steps += 1
            action = [1, 1, 1, 0, 1]  # Safe default fallback action
            step_error = "none"
            
            # --- AGENT INFERENCE BLOCK ---
            try:
                response = await client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {"role": "system", "content": "You are a factory control policy."},
                        {"role": "user", "content": PROMPT_TEMPLATE.format(obs=obs)}
                    ],
                    temperature=0.0
                )
                
                content = response.choices[0].message.content.strip()
                
                # Extract array properly to avoid string hallucinations
                start_i = content.find("[")
                end_i = content.rfind("]")
                if start_i != -1 and end_i != -1:
                    parsed = json.loads(content[start_i:end_i+1])
                    if isinstance(parsed, list) and len(parsed) == 5:
                        action = [int(a) for a in parsed]
                    else:
                        step_error = "invalid_action_format"
                else:
                    step_error = "no_json_array_found"
                    
            except Exception as e:
                step_error = f"llm_error_{type(e).__name__}"
                
            # --- ENVIRONMENT STEP BLOCK ---
            try:
                # Handle potential argument unpacking based on standard gym implementations
                if hasattr(env, 'step') and callable(getattr(env, 'step')):
                    obs, reward, terminated, truncated, info = await env.step(action)
                    done = terminated or truncated
                    step_rewards.append(reward)
                else:
                    raise RuntimeError("Environment step function invalid")
                    
                action_str = json.dumps(action, separators=(',', ':'))
                print(f"[STEP] step={total_steps} action={action_str} reward={reward:.2f} done={str(done).lower()} error={step_error}")
                
            except Exception as e:
                step_error = f"env_step_error_{type(e).__name__}"
                step_rewards.append(0.0)
                action_str = json.dumps(action, separators=(',', ':'))
                print(f"[STEP] step={total_steps} action={action_str} reward=0.00 done=true error={step_error}")
                done = True
                
        # --- SCORE COMPUTATION BLOCK ---
        try:
            # Check if compute_score is an environment method or an external function
            if hasattr(env, 'compute_score') and callable(getattr(env, 'compute_score')):
                score = await getattr(env, 'compute_score')()
            elif hasattr(env, 'state') and callable(getattr(env, 'state')):
                # In standard factory OpenEnv implementation
                state = await env.state() if asyncio.iscoroutinefunction(env.state) else env.state()
                score = compute_score(state)
            else:
                score = 0.0
                
            # Clamp to [0,1]
            score = max(0.0, min(float(score), 1.0))
            success = True
            
        except Exception as e:
            pass
            
    except Exception as e:
        # Catch unexpected global fatals (e.g., Docker crash) to ensure [END] is always printed
        pass
        
    finally:
        # 4. Mandatory completion log ensuring lowercase booleans and 2-decimal formatting
        rewards_str = ",".join(f"{r:.2f}" for r in step_rewards)
        print(f"[END] success={str(success).lower()} steps={total_steps} score={score:.2f} rewards={rewards_str}")

if __name__ == "__main__":
    asyncio.run(run_inference())
