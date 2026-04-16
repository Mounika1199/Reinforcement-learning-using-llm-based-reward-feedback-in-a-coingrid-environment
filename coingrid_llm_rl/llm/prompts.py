"""
Prompt builders for LLM-based reward evaluation.

build_episodic_prompt
    Used by HybridRewardWrapper and LLMEpisodicRewardWrapper.
    Evaluates the full episode from the summary only.
    LLM returns a score tagged ``Score: <float>``.

build_step_prompt
    Used by LLMStepwiseRewardWrapper at each step.
    Gives the LLM raw counts, agent movement, and distance-to-coin
    context.  The LLM decides a reward following open-ended guidelines.
    LLM returns a score tagged ``FScore: <float>``.

build_stepwise_episode_prompt
    Used by LLMStepwiseRewardWrapper at episode end.
    Evaluates the whole episode using the final summary plus the
    accumulated step-event log.
    LLM returns a score tagged ``FScore: <float>``.
"""

from __future__ import annotations


# ---------------------------------------------------------------------------
# Episodic prompt  (HybridRewardWrapper + LLMEpisodicRewardWrapper)
# ---------------------------------------------------------------------------

def build_episodic_prompt(instruction: str, summary: str) -> str:
    """
    Build an end-of-episode evaluation prompt.

    Scoring formula
    ~~~~~~~~~~~~~~~
    score = 1.0 − 0.5 × total_missed − 0.25 × total_extra,
    clipped to [−1.0, 1.0].

    Response tag: ``Score: <float>``
    """
    return (
        "You are an automated evaluator for a coin-collection RL task.\n\n"

        "TASK:\n"
        "Evaluate the agent's performance and compute a numeric score.\n\n"

        "SCORING RULES:\n"
        "- Start with a base score of 1.0.\n"
        "- For every missed coin, subtract 0.5.\n"
        "- For every extra or wrong-colour coin, subtract 0.25.\n"
        "- For each colour in the instruction:\n"
        "    If collected < required  →  missed = required − collected\n"
        "    If collected > required  →  extra  = collected − required\n"
        "- For each colour not in the instruction but collected  →  extra = collected\n"
        "- Final score = 1.0 − 0.5 × total_missed − 0.25 × total_extra\n"
        "- Clip the final score to [−1.0, 1.0].\n\n"

        "PROCESS:\n"
        "1. Extract all colours and counts from the instruction and agent action.\n"
        "   Include colours with count = 0.\n"
        "2. Compute missed and extra coins once per colour (no double counting).\n"
        "3. Summarise total_missed and total_extra.\n"
        "4. Compute the final score step by step.\n"
        "5. Output only in the required format.\n\n"

        "FORMAT STRICTLY:\n"
        "Explanation: <your reasoning>\n"
        "Score: <float between -1.0 and 1.0>\n\n"

        f"Instruction: {instruction}\n"
        f"Agent's action: {summary}\n"
    )


# ---------------------------------------------------------------------------
# Step prompt  (LLMStepwiseRewardWrapper — per-step call)
# ---------------------------------------------------------------------------

def build_step_prompt(
    instruction: str,
    prev_counts: dict[str, int],
    new_counts: dict[str, int],
    just_collected: list[str],
    prev_pos: tuple[int, int],
    new_pos: tuple[int, int],
    required: dict[str, int],
    coin_positions: dict[str, list],
    prev_dist: float | None,
    new_dist: float | None,
) -> str:
    """
    Build a per-step evaluation prompt.

    All numeric facts are pre-computed by Python and handed to the LLM as
    context.  The LLM follows open-ended guidelines to decide a reward in
    [-1.0, 1.0].

    Parameters
    ----------
    instruction : str
        Natural-language task instruction.
    prev_counts : dict
        Coin counts collected before this step.
    new_counts : dict
        Coin counts collected after this step.
    just_collected : list[str]
        Colours of coins picked up on this exact step.
    prev_pos, new_pos : tuple[int, int]
        Agent grid position before and after the action.
    required : dict[str, int]
        Full set of coins required by the instruction.
    coin_positions : dict[str, list]
        Initial grid positions of all coins (snapshot at episode start).
    prev_dist, new_dist : float | None
        Manhattan distance to the nearest required coin before/after the
        step.  ``None`` when no required coins remain on the board.

    Response tag: ``FScore: <float>``
    """
    return f"""
You are evaluating an RL agent step-by-step in a coin collection grid world.

Instruction:
{instruction}

Previous collected counts:
{prev_counts}

New collected counts after this step:
{new_counts}

Coins collected *on this exact step*:
{just_collected}

=== TASK ===
{instruction}

=== REQUIRED COINS ===
{required}

=== COIN POSITIONS ===
{coin_positions}

=== AGENT MOVE ===
Previous position: {prev_pos}
New position: {new_pos}

Distance to nearest REQUIRED coin before move: {prev_dist}
Distance after move: {new_dist}

=== GUIDELINES FOR REWARD ===
1. If a required coin is collected, give a positive reward.
2. If a wrong/extra coin is collected, give a negative reward.
3. If no coin collected:
   - If agent moved closer to nearest required coin (new_dist < prev_dist) → small positive reward
   - If agent moved farther from nearest required coin (new_dist > prev_dist) → small negative reward
   - If distance did not change → neutral reward
4. Output only the numeric reward as a float.

=== FORMAT ===
FScore: <float>
"""


# ---------------------------------------------------------------------------
# Stepwise episode prompt  (LLMStepwiseRewardWrapper — episode-end call)
# ---------------------------------------------------------------------------

def build_stepwise_episode_prompt(
    instruction: str,
    summary: str,
    step_events: list[str],
) -> str:
    """
    Build an end-of-episode evaluation prompt for the stepwise wrapper.

    Includes the full per-step event log so the LLM can audit the
    trajectory before assigning a final score.

    Scoring formula
    ~~~~~~~~~~~~~~~
    score = 1.0 − 0.5 × total_missed − 0.25 × total_extra,
    clipped to [−1.0, 1.0].

    Response tag: ``FScore: <float>``
    """
    return f"""
Evaluate the final episode.

Instruction:
{instruction}

Final summary:
{summary}

Step events:
{step_events}

You are an automated evaluator for a coin-collection RL task.

TASK:
Evaluate the agent's performance and compute a numeric score.

SCORING RULES:
- Start with a base score of 1.0.
- For every missed required coin, subtract 0.5.
- For every extra or wrong-colour coin, subtract 0.25.
- For each colour in the instruction:
    If collected < required  →  missed = required − collected
    If collected > required  →  extra  = collected − required
- For each colour not in the instruction but collected  →  extra = collected
- Final score = 1.0 − 0.5 × total_missed − 0.25 × total_extra
- Clip the final score to [−1.0, 1.0].

PROCESS:
1. Extract all colours and counts from the instruction and summary.
2. Compute missed and extra coins once per colour.
3. Summarise total_missed and total_extra.
4. Compute final score strictly using the formula.
5. Do not include unnecessary commentary.

FORMAT STRICTLY:
FScore: <float between -1.0 and 1.0>
"""
