"""
LLMEpisodicRewardWrapper
------------------------
Purely episodic LLM reward — the agent receives 0.0 at every step and a
single LLM-computed score only when the episode terminates.

This is the simplest LLM reward mode and is used in curriculum stage "4-llm"
as the final fine-tuning step after the agent has already acquired strong
behaviour from earlier stages.
"""

from __future__ import annotations

import re

import gymnasium as gym

from coingrid_llm_rl.llm.client import query_ollama
from coingrid_llm_rl.llm.prompts import build_episodic_prompt


class LLMEpisodicRewardWrapper(gym.Wrapper):
    """
    Wrapper that assigns reward only at episode end via an LLM call.

    Parameters
    ----------
    env : gym.Env
        Wrapped ``CoinGridEnv`` instance.
    """

    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        self.ep_rwd: list[float] = []

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.instruction = self.env.instruction
        return obs, info

    def step(self, action: int):
        obs, _, done, truncated, info = self.env.step(action)

        if done:
            summary = self.env.get_episode_summary()
            prompt = build_episodic_prompt(self.instruction, summary)
            print("[LLM-Episodic] Querying LLM …")
            response = query_ollama(prompt)
            match = re.search(r"Score:\s*(-?\d+(?:\.\d+)?)", response)
            reward = float(match.group(1)) if match else 0.0
            info["llm_score"] = reward
            self.ep_rwd.append(reward)
            print(
                f"[LLM-Episodic] score={reward:.3f} | "
                f"instruction='{self.instruction}' | {summary}"
            )
        else:
            reward = 0.0

        return obs, reward, done, truncated, info
