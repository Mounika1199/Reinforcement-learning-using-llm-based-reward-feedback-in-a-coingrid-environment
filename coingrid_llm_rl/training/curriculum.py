"""
Curriculum training pipeline.

The curriculum progresses through 10 stages, gradually increasing environment
randomness and instruction diversity while transitioning from pure Python
rewards to LLM-based rewards.

Curriculum stages
~~~~~~~~~~~~~~~~~
  Stage 1    fixed layout  + fixed instruction       (StepwiseReward)
  Stage 1.1  fixed layout  + templated 1-colour      (StepwiseReward)
  Stage 1.2  fixed layout  + templated 2-colour      (StepwiseReward)
  Stage 1.3  fixed layout  + templated 3-colour      (StepwiseReward)
  Stage 2    fixed layout  + random instruction      (StepwiseReward)
  Stage 2.1  3-fixed layout + fixed instruction      (StepwiseReward)
  Stage 2.2  1-fixed layout + fixed instruction      (StepwiseReward)
  Stage 3.5-bridge  1-fixed + varied instructions   (StepwiseReward)
  Stage 3.8-hybrid  1-fixed + varied instructions   (HybridReward)
  Stage 4-llm       1-fixed + varied instructions   (LLMEpisodicReward)
"""

from __future__ import annotations

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize

from coingrid_llm_rl.env.coingrid_env import CoinGridEnv
from coingrid_llm_rl.wrappers.stepwise import StepwiseRewardWrapper
from coingrid_llm_rl.wrappers.hybrid import HybridRewardWrapper
from coingrid_llm_rl.wrappers.llm_episodic import LLMEpisodicRewardWrapper


# ---------------------------------------------------------------------------
# Stage configuration
# ---------------------------------------------------------------------------

# Maps each stage identifier to (layout_mode, instruction_mode, varied_instructions, wrapper_class)
_STAGE_CONFIG: dict[str | float, tuple] = {
    1:            ("fixed",                "fixed",           False, StepwiseRewardWrapper),
    1.1:          ("fixed",                "templated-1color", False, StepwiseRewardWrapper),
    1.2:          ("fixed",                "templated-2color", False, StepwiseRewardWrapper),
    1.3:          ("fixed",                "templated-3color", False, StepwiseRewardWrapper),
    2:            ("fixed",                "random",           False, StepwiseRewardWrapper),
    2.1:          ("semi-random-3fixed",   "fixed",            False, StepwiseRewardWrapper),
    2.2:          ("semi-random-1fixed",   "fixed",            False, StepwiseRewardWrapper),
    "3.5-bridge": ("semi-random-1fixed",   "random",           True,  StepwiseRewardWrapper),
    "3.8-hybrid": ("semi-random-1fixed",   "random",           True,  HybridRewardWrapper),
    "4-llm":      ("semi-random-1fixed",   "random",           True,  LLMEpisodicRewardWrapper),
}

# Default timesteps per stage
STAGE_TIMESTEPS: dict[str | float, int] = {
    1:            10_000,
    1.1:          15_000,
    1.2:          20_000,
    1.3:          30_000,
    2:            50_000,
    2.1:          60_000,
    2.2:          75_000,
    "3.5-bridge": 90_000,
    "3.8-hybrid": 50_000,
    "4-llm":      10_000,
}

# Ordered list of stages for a full curriculum run
CURRICULUM_ORDER: list[str | float] = [
    1, 1.1, 1.2, 1.3, 2, 2.1, 2.2, "3.5-bridge", "3.8-hybrid", "4-llm"
]


# ---------------------------------------------------------------------------
# PPO hyper-parameters
# ---------------------------------------------------------------------------

PPO_KWARGS: dict = {
    "verbose": 1,
    "gamma": 0.99,
    "n_steps": 2048,
    "batch_size": 64,
    "gae_lambda": 0.95,
    "ent_coef": 0.01,
    "learning_rate": 3e-4,
    "n_epochs": 10,
}


# ---------------------------------------------------------------------------
# Training function
# ---------------------------------------------------------------------------

def train_curriculum_stage(
    stage: str | float,
    model: PPO | None = None,
    total_timesteps: int | None = None,
    shaping_alpha: float = 0.08,
    step_penalty: float = -0.01,
) -> tuple[PPO, list[float]]:
    """
    Train (or continue training) a PPO agent for one curriculum stage.

    Parameters
    ----------
    stage : str | float
        Stage identifier, e.g. ``1``, ``2.1``, ``"3.8-hybrid"``.
    model : PPO | None
        Existing PPO model to continue from.  If ``None`` a new model is
        created with :data:`PPO_KWARGS`.
    total_timesteps : int | None
        Steps to train for.  Defaults to :data:`STAGE_TIMESTEPS[stage]`.
    shaping_alpha : float
        Forwarded to ``StepwiseRewardWrapper`` / ``HybridRewardWrapper``.
    step_penalty : float
        Forwarded to ``StepwiseRewardWrapper`` / ``HybridRewardWrapper``.

    Returns
    -------
    model : PPO
        The trained (or updated) PPO model.
    ep_rewards : list[float]
        Per-episode reward/score list collected by the reward wrapper.
    """
    if stage not in _STAGE_CONFIG:
        raise ValueError(
            f"Unknown stage '{stage}'. Valid stages: {list(_STAGE_CONFIG)}"
        )

    layout_mode, instruction_mode, varied, WrapperCls = _STAGE_CONFIG[stage]
    timesteps = total_timesteps or STAGE_TIMESTEPS[stage]

    print(f"\n{'='*60}")
    print(f"  Stage {stage}  |  layout={layout_mode}  |  instr={instruction_mode}")
    print(f"  wrapper={WrapperCls.__name__}  |  timesteps={timesteps:,}")
    print(f"{'='*60}")

    base_env = CoinGridEnv(
        layout_mode=layout_mode,
        instruction_mode=instruction_mode,
        varied_instructions=varied,
    )

    # Wrap with appropriate reward wrapper
    if WrapperCls is StepwiseRewardWrapper:
        env = StepwiseRewardWrapper(
            base_env, shaping_alpha=shaping_alpha, step_penalty=step_penalty
        )
    elif WrapperCls is HybridRewardWrapper:
        env = HybridRewardWrapper(
            base_env, shaping_alpha=shaping_alpha, step_penalty=step_penalty
        )
    else:
        env = LLMEpisodicRewardWrapper(base_env)

    vec_env = make_vec_env(lambda: env, n_envs=1)
    vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True, clip_obs=10.0)

    if model is None:
        model = PPO("MultiInputPolicy", vec_env, **PPO_KWARGS)
    else:
        model.set_env(vec_env)

    model.learn(total_timesteps=timesteps)
    return model, env.ep_rwd


# ---------------------------------------------------------------------------
# Full curriculum runner
# ---------------------------------------------------------------------------

def run_full_curriculum(
    stages: list[str | float] | None = None,
    timesteps_override: dict[str | float, int] | None = None,
) -> tuple[PPO, dict[str | float, list[float]]]:
    """
    Run the complete curriculum end-to-end, transferring the model between
    stages.

    Parameters
    ----------
    stages : list | None
        Ordered list of stages to run.  Defaults to :data:`CURRICULUM_ORDER`.
    timesteps_override : dict | None
        Per-stage timestep overrides; missing keys fall back to
        :data:`STAGE_TIMESTEPS`.

    Returns
    -------
    model : PPO
        Final trained model.
    stagewise_rewards : dict
        Mapping of stage → per-episode reward list.
    """
    stages = stages or CURRICULUM_ORDER
    overrides = timesteps_override or {}

    stagewise_rewards: dict[str | float, list[float]] = {}
    model: PPO | None = None

    for stage in stages:
        ts = overrides.get(stage, STAGE_TIMESTEPS[stage])
        model, rewards = train_curriculum_stage(stage, model=model, total_timesteps=ts)
        stagewise_rewards[stage] = rewards

    return model, stagewise_rewards
