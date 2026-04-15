"""
StepwiseRewardWrapper
---------------------
Pure Python reward signal — no LLM queries.

Reward components per step
~~~~~~~~~~~~~~~~~~~~~~~~~~
  +0.25   for each coin collected that still satisfies the task requirement
  −0.25   for each coin collected in excess of the requirement
  −0.25   for each coin collected of a colour not in the instruction
  Δφ      potential-based shaping  φ(s) = −α·Σ needed_i·dist(agent, coin_i)
  −0.01   step penalty (encourage efficiency)

At episode end the final deterministic score is computed and logged but does
NOT override the running reward (PPO uses the running reward for learning).
"""

from __future__ import annotations

from coingrid_llm_rl.wrappers._base import BaseShapingWrapper


class StepwiseRewardWrapper(BaseShapingWrapper):
    """
    Reward wrapper that provides dense, rule-based feedback every step.

    Parameters
    ----------
    env : gym.Env
        Wrapped ``CoinGridEnv`` instance.
    shaping_alpha : float
        Potential-shaping scale factor (default 0.08).
    step_penalty : float
        Per-step time penalty (default −0.01).
    """

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.instruction = self.env.instruction
        self._init_shaping(self.instruction)
        return obs, info

    def step(self, action: int):
        prev_counts = dict(self.collected)
        prev_pot = self.prev_potential

        obs, _, done, truncated, info = self.env.step(action)

        # Rebuild collected counts from environment history
        self.collected = {}
        for color, _ in self.env.collected:
            self.collected[color] = self.collected.get(color, 0) + 1

        # ── Coin-collection reward ────────────────────────────────────────
        reward = 0.0
        for color, now in self.collected.items():
            prev = prev_counts.get(color, 0)
            for nth in range(prev + 1, now + 1):          # iterate each new coin
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

        info["stepwise_reward"] = reward

        # ── Episode end: compute & log deterministic score ────────────────
        if done:
            final_score = self._compute_final_score()
            self.ep_rwd.append(final_score)
            info["final_score"] = final_score
            print(
                f"[Stepwise] score={final_score:.3f} | "
                f"instruction='{self.instruction}' | "
                f"{self.env.get_episode_summary()}"
            )

        return obs, reward, done, truncated, info
