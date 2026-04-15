"""
coingrid_llm_rl
===============
Reinforcement learning with LLM-based reward feedback in a CoinGrid environment.

Quick start
-----------
    from coingrid_llm_rl.env.coingrid_env import CoinGridEnv
    from coingrid_llm_rl.wrappers.stepwise import StepwiseRewardWrapper
    from coingrid_llm_rl.training.curriculum import run_full_curriculum

    model, rewards = run_full_curriculum()
"""
