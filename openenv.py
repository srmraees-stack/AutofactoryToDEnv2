import asyncio
from server.environment import AutoFactoryToDEnv, compute_score as _compute_score

class MyEnv:
    """Async wrapper for AutoFactoryToDEnv to support inference.py testing."""
    
    def __init__(self, env=None):
        self._env = env or AutoFactoryToDEnv()
        
    @classmethod
    async def from_docker_image(cls, image_name: str):
        """Mock implementation: returns a local environment instance."""
        return cls()
        
    async def reset(self):
        """Async wrapper for env.reset()."""
        obs, info = self._env.reset()
        return obs, info
        
    async def step(self, action):
        """Async wrapper for env.step(action)."""
        # action is expected to be a list/tuple of 5 integers
        obs, reward, terminated, truncated, info = self._env.step(*action)
        return obs, reward, terminated, truncated, info
        
    async def state(self):
        """Async wrapper for env.state()."""
        return self._env.state()
        
    async def compute_score(self):
        """Async wrapper for computing the final score."""
        return _compute_score(self._env.state())

def compute_score(state):
    """Module-level compute_score function exported for compatibility."""
    return _compute_score(state)
