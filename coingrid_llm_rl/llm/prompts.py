"""
Prompt builders for LLM-based reward evaluation.

Two prompt styles are provided:

build_episodic_prompt
    Used by HybridRewardWrapper and LLMEpisodicRewardWrapper.
    Evaluates the full episode from the final summary only.
    LLM returns a score tagged ``Score: <float>``.

build_step_prompt
    Used by LLMStepwiseRewardWrapper.
    Evaluates a single agent action from pre-computed facts (position
    change, collected-coin classification, distance trend).
    LLM returns a score tagged ``Score: <float>``.

All numeric facts (distances, coin counts, classifications) are computed
in Python and provided to the LLM as structured context so the model only
needs to apply a fixed decision rule — not compute arithmetic itself.
"""

from __future__ import annotations


# ---------------------------------------------------------------------------
# Episodic prompt  (Hybrid + LLMEpisodic)
# ---------------------------------------------------------------------------

def build_episodic_prompt(instruction: str, summary: str) -> str:
    """
    Build an evaluation prompt for end-of-episode scoring.

    The LLM is given the task instruction and the collected-coin summary,
    applies the fixed scoring formula, and returns ``Score: <float>``.

    Scoring formula
    ~~~~~~~~~~~~~~~
    score = 1.0 − 0.5 × total_missed − 0.25 × total_extra
    clipped to [−1.0, 1.0]

    Parameters
    ----------
    instruction : str
        Natural-language task instruction given to the agent.
    summary : str
        Output of ``CoinGridEnv.get_episode_summary()``.

    Returns
    -------
    str
        Fully-formatted prompt string ready to send to the LLM.
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
# Stepwise prompt  (LLMStepwiseRewardWrapper)
# ---------------------------------------------------------------------------

def build_step_prompt(
    instruction: str,
    prev_pos: tuple[int, int],
    new_pos: tuple[int, int],
    collected_info: list[dict],
    remaining_required: dict[str, int],
    remaining_positions: dict[str, list],
    distance_trend: str,
) -> str:
    """
    Build an evaluation prompt for a single agent step.

    All numeric facts are pre-computed by Python; the LLM applies a
    deterministic decision rule and returns ``Score: <float>``.

    Decision rule
    ~~~~~~~~~~~~~
    1. If any collected coin is classified ``"extra"``    →  Score = −0.35
    2. Else if any collected coin is classified ``"required"`` →  Score = +0.35
    3. Else (no coin collected):
        distance_trend == "closer"      →  Score = +0.10
        distance_trend == "farther"     →  Score = −0.10
        distance_trend == "same"        →  Score = −0.025
        distance_trend == "no_required" →  Score =  0.0

    Parameters
    ----------
    instruction : str
        Natural-language task instruction.
    prev_pos, new_pos : tuple[int, int]
        Agent grid position before and after the action.
    collected_info : list[dict]
        Each entry is ``{"color": str, "type": "required" | "extra"}``.
    remaining_required : dict[str, int]
        Coins still needed to complete the task.
    remaining_positions : dict[str, list]
        Grid positions of coins still on the board.
    distance_trend : str
        One of ``"closer"``, ``"farther"``, ``"same"``, ``"no_required"``.

    Returns
    -------
    str
        Fully-formatted prompt string ready to send to the LLM.
    """
    return (
        "You are evaluating ONE action in a grid-world coin collection task.\n\n"

        f"TASK INSTRUCTION:\n{instruction}\n\n"

        f"REMAINING REQUIRED COINS:\n{remaining_required}\n\n"

        f"REMAINING COIN POSITIONS:\n{remaining_positions}\n\n"

        f"AGENT MOVE:\n"
        f"Previous position: {prev_pos}\n"
        f"New position:      {new_pos}\n\n"

        f"COLLECTED THIS STEP:\n{collected_info}\n\n"

        f"DISTANCE TREND (system-computed):\n{distance_trend}\n\n"

        "STRICT DECISION RULE:\n"
        '1. If ANY collected coin has type == "extra"     → Score = -0.35\n'
        '2. ELSE IF ANY coin has type == "required"       → Score = +0.35\n'
        "3. ELSE (no coin collected):\n"
        '   distance_trend == "closer"      → Score = +0.10\n'
        '   distance_trend == "farther"     → Score = -0.10\n'
        '   distance_trend == "same"        → Score = -0.025\n'
        '   distance_trend == "no_required" → Score =  0.0\n\n'

        "OUTPUT FORMAT (STRICT):\n"
        "Score: <float>\n"
    )
