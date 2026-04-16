"""
LLMStepwiseRewardWrapper
-------------------------
LLM reward wrapper that evaluates every step and the full episode.

Design
~~~~~~
At each step, Python provides the LLM with:
  - the instruction and required coins
  - previous and new collected-coin counts
  - which coins were picked up this step
  - agent movement (prev/new position)
  - Manhattan distance to the nearest required coin before and after the step

The LLM applies open-ended guidelines to decide an appropriate reward in
[-1.0, 1.0] and returns it tagged as ``FScore: <float>``.

At episode end the LLM additionally receives the full step-event log and
the final summary, and assigns a terminal score using the standard formula.

Distance notes
~~~~~~~~~~~~~~
Distance is computed against ``coin_positions`` — the initial grid snapshot
taken at ``reset()`` time.  This means the distance signal may reference
positions of already-collected coins; the LLM has full context (counts,
step events) to interpret this correctly.
"""

from __future__ import annotations

import re

import gymnasium as gym

from coingrid_llm_rl.llm.client import query_ollama
from coingrid_llm_rl.llm.prompts import build_step_prompt, build_stepwise_episode_prompt


class LLMStepwiseRewardWrapper(gym.Wrapper):
    """
    Dense LLM reward wrapper: one LLM call per step + one at episode end.

    Parameters
    ----------
    env : gym.Env
        Wrapped ``CoinGridEnv`` instance.  The environment must expose
        ``required_coins`` and ``coin_positions`` after ``reset()``
        (both are provided by ``CoinGridEnv``).
    """

    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        self.collected: dict[str, int] = {}
        self.episode_step_events: list[str] = []
        self.ep_rwd: list[float] = []

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.collected = {}
        self.episode_step_events = []
        self.instruction = self.env.instruction
        self.required = self.env.required_coins       # {color: count}
        self.coin_positions = self.env.coin_positions  # {color: [(r,c), …]}
        return obs, info

    # ------------------------------------------------------------------
    # Step
    # ------------------------------------------------------------------

    def step(self, action: int):
        prev_pos = tuple(self.env.agent_pos)
        prev_counts = dict(self.collected)

        # ── Environment step ─────────────────────────────────────────────
        obs, _, done, truncated, info = self.env.step(action)
        new_pos = tuple(self.env.agent_pos)

        # ── Update collected counts ───────────────────────────────────────
        self.collected = {}
        for color, _ in self.env.collected:
            self.collected[color] = self.collected.get(color, 0) + 1

        # Coins picked up on this exact step
        just_collected = [
            color
            for color, new_ct in self.collected.items()
            for _ in range(new_ct - prev_counts.get(color, 0))
        ]

        # ── Distance to nearest required coin (from initial positions) ────
        required_locations = [
            pos
            for col, needed in self.required.items()
            if needed > 0 and col in self.coin_positions
            for pos in self.coin_positions[col]
        ]
        if required_locations:
            prev_dist = min(self._manhattan(prev_pos, p) for p in required_locations)
            new_dist = min(self._manhattan(new_pos, p) for p in required_locations)
        else:
            prev_dist = new_dist = 0

        # ── Query LLM for step reward ─────────────────────────────────────
        step_prompt = build_step_prompt(
            instruction=self.instruction,
            prev_counts=prev_counts,
            new_counts=self.collected,
            just_collected=just_collected,
            prev_pos=prev_pos,
            new_pos=new_pos,
            required=self.required,
            coin_positions=self.coin_positions,
            prev_dist=prev_dist,
            new_dist=new_dist,
        )
        response = query_ollama(step_prompt)
        scores = re.findall(r"FScore:\s*(-?\d+(?:\.\d+)?)", response)
        step_reward = max(-1.0, min(1.0, float(scores[-1]))) if scores else 0.0

        print(
            f"[LLM-Step] {prev_pos}→{new_pos} | "
            f"collected={just_collected} | "
            f"dist {prev_dist}→{new_dist} | reward={step_reward:.3f}"
        )
        self.episode_step_events.append(
            f"Step: {prev_pos}->{new_pos}, "
            f"collected={just_collected}, reward={step_reward:.3f}"
        )
        info["llm_step_reward"] = step_reward

        # ── Episode end: final episodic LLM score ────────────────────────
        if done:
            summary = self.env.get_episode_summary()
            final_prompt = build_stepwise_episode_prompt(
                self.instruction, summary, self.episode_step_events
            )
            final_resp = query_ollama(final_prompt)
            final_scores = re.findall(r"FScore:\s*(-?\d+(?:\.\d+)?)", final_resp)
            final_reward = (
                max(-1.0, min(1.0, float(final_scores[-1])))
                if final_scores else 0.0
            )
            info["llm_final_reward"] = final_reward
            self.ep_rwd.append(final_reward)
            print(
                f"\n[LLM-Step] EPISODE END | "
                f"instruction='{self.instruction}' | {summary}\n"
                f"Final reward: {final_reward:.3f}"
            )
            return obs, final_reward, done, truncated, info

        return obs, step_reward, done, truncated, info

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _manhattan(a: tuple[int, int], b: tuple[int, int]) -> int:
        return abs(a[0] - b[0]) + abs(a[1] - b[1])
