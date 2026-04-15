"""
LLMStepwiseRewardWrapper
-------------------------
Most sophisticated LLM reward wrapper.

Design principle
~~~~~~~~~~~~~~~~
**Python computes every numeric fact; the LLM applies a symbolic rule.**

At each step Python pre-computes:
  - which coins were collected and whether they count as "required" or "extra"
  - the Manhattan-distance trend (closer / farther / same / no_required)

These facts are handed to the LLM in a structured prompt.  The model follows
a strict, unambiguous decision rule and returns a single float.

At episode end, the LLM additionally receives the full episode summary and
applies the standard scoring formula for a final episodic score.

Bug-fix vs. original notebook
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The original code classified collected coins **after** incrementing
``required_collected``, which caused valid coins to be mis-labelled as
"extra" when the required count was exactly 1.  The fix is to record the
pre-increment count and classify against that snapshot.
"""

from __future__ import annotations

import re

import gymnasium as gym

from coingrid_llm_rl.llm.client import query_ollama
from coingrid_llm_rl.llm.prompts import build_step_prompt, build_episodic_prompt


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
        self.required_collected: dict[str, int] = {}   # task-useful coins only
        self.episode_step_events: list[str] = []
        self.ep_rwd: list[float] = []

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.collected = {}
        self.required_collected = {}
        self.episode_step_events = []
        self.instruction = self.env.instruction
        self.required = self.env.required_coins          # {color: count}
        self.coin_positions = self.env.coin_positions    # {color: [(r,c), …]}
        return obs, info

    # ------------------------------------------------------------------
    # Step
    # ------------------------------------------------------------------

    def step(self, action: int):
        prev_pos = tuple(self.env.agent_pos)

        # ── Snapshot state before the environment step ────────────────────
        remaining_before = self._remaining_required()
        remaining_pos_before = self._remaining_coin_positions()
        required_locs = [
            pos
            for color, n in remaining_before.items()
            if n > 0
            for pos in self.coin_positions.get(color, [])
        ]
        prev_dist = (
            min(self._manhattan(prev_pos, p) for p in required_locs)
            if required_locs else None
        )
        prev_collected = dict(self.collected)

        # ── Environment step ─────────────────────────────────────────────
        obs, _, done, truncated, info = self.env.step(action)
        new_pos = tuple(self.env.agent_pos)

        # ── Update raw collected counts ───────────────────────────────────
        self.collected = {}
        for color, _ in self.env.collected:
            self.collected[color] = self.collected.get(color, 0) + 1

        just_collected = [
            color
            for color, new_ct in self.collected.items()
            for _ in range(new_ct - prev_collected.get(color, 0))
        ]

        # ── Classify coins BEFORE updating required_collected (bug-fix) ───
        collected_info = []
        for color in just_collected:
            req_so_far = self.required_collected.get(color, 0)
            if color in self.required and req_so_far < self.required[color]:
                collected_info.append({"color": color, "type": "required"})
                self.required_collected[color] = req_so_far + 1
            else:
                collected_info.append({"color": color, "type": "extra"})

        # ── Distance after step (same target locations as before) ─────────
        new_dist = (
            min(self._manhattan(new_pos, p) for p in required_locs)
            if required_locs else None
        )

        # ── Distance trend (computed in Python, not by LLM) ──────────────
        if prev_dist is None:
            distance_trend = "no_required"
        elif new_dist < prev_dist:
            distance_trend = "closer"
        elif new_dist > prev_dist:
            distance_trend = "farther"
        else:
            distance_trend = "same"

        # ── Query LLM for step reward ─────────────────────────────────────
        step_prompt = build_step_prompt(
            instruction=self.instruction,
            prev_pos=prev_pos,
            new_pos=new_pos,
            collected_info=collected_info,
            remaining_required=remaining_before,
            remaining_positions=remaining_pos_before,
            distance_trend=distance_trend,
        )
        response = query_ollama(step_prompt)
        scores = re.findall(r"Score:\s*(-?\d+(?:\.\d+)?)", response)
        step_reward = max(-1.0, min(1.0, float(scores[-1]))) if scores else 0.0

        print(
            f"[LLM-Step] {prev_pos}→{new_pos} | "
            f"collected={collected_info} | trend={distance_trend} | "
            f"reward={step_reward:.3f}"
        )
        self.episode_step_events.append(
            f"{prev_pos}->{new_pos}, collected={collected_info}, "
            f"trend={distance_trend}, reward={step_reward:.3f}"
        )
        info["llm_step_reward"] = step_reward

        # ── Episode end: final episodic LLM score ────────────────────────
        if done:
            summary = self.env.get_episode_summary()
            final_prompt = build_episodic_prompt(self.instruction, summary)
            final_resp = query_ollama(final_prompt)
            final_scores = re.findall(r"Score:\s*(-?\d+(?:\.\d+)?)", final_resp)
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

    def _remaining_required(self) -> dict[str, int]:
        """Coins still needed to complete the task."""
        return {
            color: needed - self.required_collected.get(color, 0)
            for color, needed in self.required.items()
            if needed - self.required_collected.get(color, 0) > 0
        }

    def _remaining_coin_positions(self) -> dict[str, list[tuple[int, int]]]:
        """Positions of coins still present on the grid."""
        collected_positions = {pos for _, pos in self.env.collected}
        return {
            color: [p for p in positions if p not in collected_positions]
            for color, positions in self.coin_positions.items()
            if any(p not in collected_positions for p in positions)
        }

    @staticmethod
    def _manhattan(a: tuple[int, int], b: tuple[int, int]) -> int:
        return abs(a[0] - b[0]) + abs(a[1] - b[1])
