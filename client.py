"""
client.py — HTTP client for AutoFactoryToDEnv server.

Usage
-----
    from client import FactoryEnvClient

    client = FactoryEnvClient(base_url="http://localhost:8000")

    obs, info = client.reset()
    obs, reward, terminated, truncated, info = client.step(
        stamping=2, molding=2, cnc=2, compressor=1, welder=0
    )
"""

import requests
from typing import Any, Dict, Tuple


class FactoryEnvClient:
    """Thin HTTP wrapper around the AutoFactoryToDEnv FastAPI server."""

    def __init__(self, base_url: str = "http://localhost:8000") -> None:
        self.base_url = base_url.rstrip("/")

    # ------------------------------------------------------------------
    # Public API (mirrors the gym-style interface)
    # ------------------------------------------------------------------

    def reset(self) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Reset the server-side environment.

        Returns
        -------
        observation : dict
        info        : dict
        """
        resp = self._post("/reset", json={})
        data = resp.json()
        return data["observation"], data["info"]

    def step(
        self,
        stamping:   int,
        molding:    int,
        cnc:        int,
        compressor: int,
        welder:     int,
    ) -> Tuple[Dict[str, Any], float, bool, bool, Dict[str, Any]]:
        """
        Send one action to the server.

        Returns
        -------
        observation : dict
        reward      : float
        terminated  : bool
        truncated   : bool
        info        : dict
        """
        payload = {
            "stamping":   stamping,
            "molding":    molding,
            "cnc":        cnc,
            "compressor": compressor,
            "welder":     welder,
        }
        resp = self._post("/step", json=payload)
        data = resp.json()
        return (
            data["observation"],
            data["reward"],
            data["terminated"],
            data["truncated"],
            data["info"],
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _post(self, path: str, json: dict) -> requests.Response:
        url  = f"{self.base_url}{path}"
        resp = requests.post(url, json=json, timeout=10)
        if not resp.ok:
            raise RuntimeError(
                f"Server error {resp.status_code} on {path}: {resp.text}"
            )
        return resp


# ---------------------------------------------------------------------------
# Quick smoke-test (run: python client.py)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    client = FactoryEnvClient()

    print("=== AutoFactoryToDEnv smoke-test ===\n")

    obs, info = client.reset()
    print(f"[reset]  obs={obs}\n         info={info}\n")

    total_reward = 0.0
    for hour in range(24):
        # Greedy policy: run everything at full power all the time
        obs, reward, terminated, truncated, info = client.step(
            stamping=2, molding=2, cnc=2, compressor=1, welder=1
        )
        total_reward += reward
        print(
            f"[step {hour+1:02d}] reward={reward:+.4f}  "
            f"prod={obs['production_so_far']:.0f}  "
            f"health={[f'{h:.2f}' for h in obs['machine_health']]}"
        )
        if terminated:
            target_met = info.get("target_met")
            print(f"\nEpisode done. Target met: {target_met}. Total reward: {total_reward:.4f}")
            break
