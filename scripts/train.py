"""
CLI entry point for running the CoinGrid LLM-RL curriculum.

Usage examples
--------------
# Full curriculum with default timesteps
python scripts/train.py

# Run only the LLM-reward stages
python scripts/train.py --stages 3.5-bridge 3.8-hybrid 4-llm

# Override timesteps for specific stages
python scripts/train.py --stages 1 2 --timesteps 1:5000 2:20000

# Save the final model
python scripts/train.py --save-model models/final_ppo
"""

from __future__ import annotations

import argparse
import sys

from coingrid_llm_rl.training.curriculum import (
    run_full_curriculum,
    CURRICULUM_ORDER,
    STAGE_TIMESTEPS,
)
from coingrid_llm_rl.training.plotting import plot_stagewise_rewards


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a PPO agent on CoinGrid with LLM-based reward feedback."
    )
    parser.add_argument(
        "--stages",
        nargs="+",
        default=None,
        help=(
            "Ordered list of stages to run.  "
            "Numeric stages: 1 1.1 1.2 1.3 2 2.1 2.2.  "
            "String stages: 3.5-bridge 3.8-hybrid 4-llm.  "
            "Default: full curriculum."
        ),
    )
    parser.add_argument(
        "--timesteps",
        nargs="*",
        default=[],
        metavar="STAGE:N",
        help=(
            "Per-stage timestep overrides in 'STAGE:N' format, e.g. "
            "'1:5000 2:20000 4-llm:5000'."
        ),
    )
    parser.add_argument(
        "--save-model",
        default=None,
        metavar="PATH",
        help="Save the final PPO model to this path (e.g. models/final_ppo).",
    )
    parser.add_argument(
        "--plot-stages",
        nargs="*",
        default=None,
        metavar="STAGE",
        help=(
            "Stages to highlight in the reward plot.  "
            "Default: 3.5-bridge 3.8-hybrid 4-llm."
        ),
    )
    parser.add_argument(
        "--save-plot",
        default=None,
        metavar="PATH",
        help="Save the reward plot to this file path (e.g. results/plot.png).",
    )
    return parser.parse_args(argv)


def _coerce_stage(s: str) -> str | float:
    """Convert a stage string to float if possible, else keep as str."""
    try:
        return float(s)
    except ValueError:
        return s


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    # ── Stage list ────────────────────────────────────────────────────────
    stages = (
        [_coerce_stage(s) for s in args.stages]
        if args.stages
        else CURRICULUM_ORDER
    )

    # ── Timestep overrides ────────────────────────────────────────────────
    overrides: dict = {}
    for item in args.timesteps or []:
        stage_str, n_str = item.split(":")
        overrides[_coerce_stage(stage_str)] = int(n_str)

    # ── Train ─────────────────────────────────────────────────────────────
    model, stagewise_rewards = run_full_curriculum(
        stages=stages,
        timesteps_override=overrides,
    )

    # ── Save model ────────────────────────────────────────────────────────
    if args.save_model:
        model.save(args.save_model)
        print(f"\nModel saved to {args.save_model}")

    # ── Plot ──────────────────────────────────────────────────────────────
    highlight = (
        [_coerce_stage(s) for s in args.plot_stages]
        if args.plot_stages is not None
        else ["3.5-bridge", "3.8-hybrid", "4-llm"]
    )
    plot_stagewise_rewards(
        stagewise_rewards,
        selected_stages=highlight,
        save_path=args.save_plot,
    )


if __name__ == "__main__":
    main(sys.argv[1:])
