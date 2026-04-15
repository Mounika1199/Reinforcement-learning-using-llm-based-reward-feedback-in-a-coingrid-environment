"""
Base reward-shaping wrapper shared by StepwiseRewardWrapper and
HybridRewardWrapper.

Provides:
  - instruction parsing
  - potential-based reward shaping (distance to nearest required coin)
  - final episode score computation
"""

from __future__ import annotations

import gymnasium as gym

from coingrid_llm_rl.env.coingrid_env import parse_instruction


class BaseShapingWrapper(gym.Wrapper):
    """
    Abstract base for wrappers that use potential-based reward shaping.

    Subclasses must call ``super().__init__`` and should call
    ``self._init_shaping()`` in their own ``reset()`` after calling
    ``super().reset()``.

    Parameters
    ----------
    env : gym.Env
        The environment to wrap (must be a ``CoinGridEnv`` or compatible).
    shaping_alpha : float
        Scaling factor for the potential function.  Higher values produce
        stronger distance-shaping signals.
    step_penalty : float
        Small negative reward added every step to encourage efficiency.
    """

    def __init__(
        self,
        env: gym.Env,
        shaping_alpha: float = 0.08,
        step_penalty: float = -0.01,
    ) -> None:
        super().__init__(env)
        self.shaping_alpha = shaping_alpha
        self.step_penalty = step_penalty

        self.required: dict[str, int] = {}
        self.collected: dict[str, int] = {}
        self.prev_potential: float = 0.0
        self.ep_rwd: list[float] = []

    # ------------------------------------------------------------------
    # Shaping state – call from subclass reset()
    # ------------------------------------------------------------------

    def _init_shaping(self, instruction: str) -> None:
        self.required = parse_instruction(instruction)
        self.collected = {}
        self.prev_potential = self._potential()

    # ------------------------------------------------------------------
    # Potential function  φ(s) = −α · Σ needed_i · dist(agent, nearest_i)
    # ------------------------------------------------------------------

    def _potential(self) -> float:
        total = 0.0
        for color, req in self.required.items():
            needed = req - self.collected.get(color, 0)
            if needed > 0:
                dist = self._nearest_distance_of_color(color)
                if dist is None:
                    dist = self.env.grid_size * 2  # fallback when coin absent
                total += needed * dist
        return -self.shaping_alpha * total

    def _nearest_distance_of_color(self, color: str) -> int | None:
        """Manhattan distance from agent to the nearest coin of ``color``."""
        color_idx = self.env.COLORS.index(color) + 1
        ax, ay = self.env.agent_pos
        min_dist: int | None = None
        for x in range(self.env.grid_size):
            for y in range(self.env.grid_size):
                if self.env.grid[x, y] == color_idx:
                    d = abs(ax - x) + abs(ay - y)
                    if min_dist is None or d < min_dist:
                        min_dist = d
        return min_dist

    # ------------------------------------------------------------------
    # Evaluation score (used at episode end for logging / analysis)
    # ------------------------------------------------------------------

    def _compute_final_score(self) -> float:
        """
        Deterministic episode score.

        score = 1.0 − 0.5 × missed − 0.25 × extra,  clipped to [−1, 1].
        """
        missed = 0
        extra = 0

        for color, req in self.required.items():
            got = self.collected.get(color, 0)
            if got < req:
                missed += req - got
            elif got > req:
                extra += got - req

        for color, got in self.collected.items():
            if color not in self.required:
                extra += got

        score = 1.0 - 0.5 * missed - 0.25 * extra
        return max(-1.0, min(1.0, score))
