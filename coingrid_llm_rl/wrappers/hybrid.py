"""
HybridRewardWrapper
-------------------
Combines dense Python reward shaping (every step) with a sparse LLM score
added at episode end.

Reward structure
~~~~~~~~~~~~~~~~
  Every step  → same coin + shaping + penalty as StepwiseRewardWrapper
  Episode end → running_reward  +  llm_weight × LLM_score

The LLM score is added on top of (not replacing) the accumulated reward so
that the agent still benefits from the dense per-step guidance.
"""

from __future__ import annotations

import re

import gymnasium as gym

from coingrid_llm_rl.llm.client import query_ollama
from coingrid_llm_rl.llm.prompts import build_episodic_prompt
from coingrid_llm_rl.wrappers._base import BaseShapingWrapper


class HybridRewardWrapper(BaseShapingWrapper):
    """
    Dense Python shaping every step + sparse LLM reward at episode end.

    Parameters
    ----------
    env : gym.Env
        Wrapped ``CoinGridEnv`` instance.
    shaping_alpha : float
        Potential-shaping scale factor (default 0.08).
    step_penalty : float
        Per-step time penalty (default −0.01).
    llm_weight : float
        Scalar multiplied with the LLM score before adding to the final
        step's reward (default 1.0).
    """

    def __init__(
        self,
        env: gym.Env,
        shaping_alpha: float = 0.08,
        step_penalty: float = -0.01,
        llm_weight: float = 1.0,
    ) -> None:
        super().__init__(env, shaping_alpha=shaping_alpha, step_penalty=step_penalty)
        self.llm_weight = llm_weight

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.instruction = self.env.instruction
        self._init_shaping(self.instruction)
        return obs, info

    def step(self, action: int):
        prev_counts = dict(self.collected)
        prev_pot = self.prev_potential

        obs, _, done, truncated, info = self.env.step(action)

        # Rebuild collected counts
        self.collected = {}
        for color, _ in self.env.collected:
            self.collected[color] = self.collected.get(color, 0) + 1

        # ── Coin-collection reward ────────────────────────────────────────
        reward = 0.0
        for color, now in self.collected.items():
            prev = prev_counts.get(color, 0)
            for nth in range(prev + 1, now + 1):
                if color in self.required and nth <= self.required[color]:
                    reward += 0.25
                else:
                    reward -= 0.25

        # ── Potential-based shaping ───────────────────────────────────────
        new_pot = self._potential()
        reward += new_pot - prev_pot
        self.prev_potential = new_pot

        # ── Step penalty ─────────────────────────────────────────────────
        reward += self.step_penalty

        # ── LLM episodic reward (added only at episode end) ───────────────
        if done:
            summary = self.env.get_episode_summary()
            prompt = build_episodic_prompt(self.instruction, summary)
            print("[Hybrid] Querying LLM …")
            response = query_ollama(prompt)
            match = re.search(r"Score:\s*(-?\d+(?:\.\d+)?)", response)
            llm_score = float(match.group(1)) if match else 0.0

            reward += self.llm_weight * llm_score
            self.ep_rwd.append(llm_score)
            info["llm_score"] = llm_score
            print(
                f"[Hybrid] llm_score={llm_score:.3f} | "
                f"instruction='{self.instruction}' | {summary}"
            )

        return obs, reward, done, truncated, info
