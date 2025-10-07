from typing import Optional, Dict
from dataclasses import dataclass
from src.core.policy import CommunicationObjective
from src.utils.logging import get_logger

logger = get_logger("message_gen")


MESSAGE_TEMPLATES = {
    CommunicationObjective.REQUEST_CLARIFICATION: """Please clarify: {focus}

My current understanding: {reasoning_summary}

Specific question: {question}""",

    CommunicationObjective.PROPOSE_REFINEMENT: """I suggest refining our approach:

Current approach: {current_approach}

Proposed refinement: {refinement}

Rationale: {rationale}""",

    CommunicationObjective.HIGHLIGHT_DISCREPANCY: """I've identified a potential discrepancy:

Your perspective: {their_view}

My perspective: {my_view}

Gap: {discrepancy}""",

    CommunicationObjective.CHALLENGE_ASSUMPTION: """Questioning an assumption:

Assumption: {assumption}

Counter-evidence: {evidence}

Alternative view: {alternative}""",

    CommunicationObjective.PROVIDE_EVIDENCE: """Supporting evidence:

Claim: {claim}

Evidence: {evidence}

Implications: {implications}""",

    CommunicationObjective.SYNTHESIZE_PERSPECTIVES: """Synthesizing our perspectives:

Key points from discussion: {key_points}

Integrated view: {synthesis}

Next steps: {next_steps}""",

    CommunicationObjective.REQUEST_ELABORATION: """Please elaborate on: {topic}

Context: {context}

Specific aspect: {aspect}""",

    CommunicationObjective.SUGGEST_ALTERNATIVE: """Alternative approach:

Current direction: {current}

Proposed alternative: {alternative}

Trade-offs: {tradeoffs}""",

    CommunicationObjective.CONFIRM_UNDERSTANDING: """Confirming my understanding:

I understand that: {understanding}

Is this correct? {verification}""",

    CommunicationObjective.SIGNAL_AGREEMENT: """Agreement:

I agree with: {agreement_point}

This aligns with: {alignment}

Moving forward: {action}""",
}


@dataclass
class MessageParams:
    """Parameters for message generation"""
    objective: CommunicationObjective
    own_reasoning: str
    target_agent_id: int
    gap_info: Optional[Dict] = None
    context: Optional[str] = None


def format_message(
    objective: CommunicationObjective,
    **kwargs,
) -> str:
    """
    Format a message using templates

    Args:
        objective: Communication objective
        **kwargs: Template parameters

    Returns:
        Formatted message
    """
    template = MESSAGE_TEMPLATES.get(objective)
    if not template:
        logger.warning(f"No template for objective: {objective}")
        return f"Message for {objective.value}"

    try:
        # Fill in available kwargs, use placeholder for missing
        safe_kwargs = {k: kwargs.get(k, "[To be determined]") for k in ["focus", "reasoning_summary", "question", "current_approach", "refinement", "rationale", "their_view", "my_view", "discrepancy", "assumption", "evidence", "alternative", "claim", "implications", "key_points", "synthesis", "next_steps", "topic", "context", "aspect", "current", "tradeoffs", "understanding", "verification", "agreement_point", "alignment", "action"]}

        message = template.format(**{k: v for k, v in safe_kwargs.items() if k in template})
        return message

    except KeyError as e:
        logger.error(f"Missing template parameter: {e}")
        return f"Message for {objective.value} (parameter missing)"


def generate_message_prompt(params: MessageParams) -> str:
    """
    Generate a prompt for LLM-based message generation

    Args:
        params: Message parameters

    Returns:
        Prompt string
    """
    prompt = f"""Generate a strategic communication message with the following objective: {params.objective.value}

Your reasoning:
{params.own_reasoning}"""

    if params.gap_info:
        prompt += f"\n\nGap analysis with target agent:"
        for gap_type, score in params.gap_info.items():
            prompt += f"\n- {gap_type}: {score:.3f}"

    if params.context:
        prompt += f"\n\nAdditional context:\n{params.context}"

    prompt += f"""

Generate a concise, strategic message (2-4 sentences) that:
1. Achieves the communication objective
2. Is clear and actionable
3. Addresses identified gaps
4. Moves the collaboration forward

Message:"""

    return prompt


def extract_key_points(reasoning: str, max_points: int = 3) -> str:
    """
    Extract key points from reasoning text

    Args:
        reasoning: Reasoning text
        max_points: Maximum number of points

    Returns:
        Formatted key points
    """
    # Simple extraction based on sentences
    sentences = reasoning.split('. ')
    key_sentences = sentences[:max_points]
    return '\n'.join(f"- {s.strip()}" for s in key_sentences if s.strip())


def summarize_reasoning(reasoning: str, max_length: int = 200) -> str:
    """
    Create a brief summary of reasoning

    Args:
        reasoning: Full reasoning text
        max_length: Maximum length

    Returns:
        Summary
    """
    if len(reasoning) <= max_length:
        return reasoning

    # Take first max_length chars and find last complete sentence
    truncated = reasoning[:max_length]
    last_period = truncated.rfind('.')

    if last_period > max_length // 2:
        return truncated[:last_period + 1]
    else:
        return truncated + "..."
