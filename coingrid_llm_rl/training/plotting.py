"""
Plotting utilities for visualising curriculum training performance.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt


def plot_stagewise_rewards(
    stagewise_rewards: dict,
    selected_stages: list | None = None,
    window: int = 50,
    ylim: tuple[float, float] = (-0.2, 1.0),
    title: str = "Training Performance Across Curriculum Stages",
    figsize: tuple[int, int] = (12, 5),
    save_path: str | None = None,
) -> None:
    """
    Plot smoothed per-episode rewards for each curriculum stage.

    Stages in ``selected_stages`` are drawn with full opacity and a thicker
    line; all others are drawn with reduced opacity.  Vertical dashed lines
    mark stage boundaries.

    Parameters
    ----------
    stagewise_rewards : dict
        Mapping of stage → list[float] as returned by ``run_full_curriculum``.
    selected_stages : list | None
        Stages to highlight.  If ``None`` all stages are plotted equally.
    window : int
        Rolling-average window size (default 50).
    ylim : tuple[float, float]
        Y-axis limits (default (−0.2, 1.0)).
    title : str
        Plot title.
    figsize : tuple[int, int]
        Figure size in inches.
    save_path : str | None
        If provided, save the figure to this path instead of (or in
        addition to) displaying it.
    """
    selected_stages = selected_stages or list(stagewise_rewards.keys())

    fig, ax = plt.subplots(figsize=figsize)

    cumulative_episodes = 0
    stage_boundaries: dict = {}

    for stage, rewards in stagewise_rewards.items():
        n = len(rewards)
        episodes = list(range(1, n + 1))

        # Rolling-average smoothing
        if n >= window:
            smoothed = np.convolve(rewards, np.ones(window) / window, mode="valid")
            episodes = list(range(window, n + 1))
        else:
            smoothed = np.array(rewards)

        offset = [e + cumulative_episodes for e in episodes]
        highlighted = stage in selected_stages

        ax.plot(
            offset,
            smoothed,
            label=f"Stage {stage}",
            linewidth=2.5 if highlighted else 1.2,
            alpha=1.0 if highlighted else 0.4,
        )

        cumulative_episodes += n
        stage_boundaries[stage] = cumulative_episodes

    # Stage boundary markers
    for boundary in stage_boundaries.values():
        ax.axvline(boundary, linestyle="--", color="grey", alpha=0.45)

    ax.set_title(title, fontsize=13)
    ax.set_xlabel("Episodes")
    ax.set_ylabel("Reward / Score")
    ax.set_ylim(*ylim)
    ax.legend(fontsize=8, ncol=3)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150)
        print(f"Figure saved to {save_path}")

    plt.show()
