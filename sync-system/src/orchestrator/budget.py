"""Budget-aware decisions for adaptive collaboration."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class BudgetAction(str, Enum):
    STOP = "stop"
    SYNTHESIZE = "synthesize"
    CRITIQUE = "critique"
    ADD_AGENT = "add_agent"


@dataclass(frozen=True)
class BudgetState:
    spent_tokens: int
    token_limit: int
    rounds: int
    round_limit: int
    uncertainty: float
    disagreement: float

    @property
    def remaining_fraction(self) -> float:
        if self.token_limit <= 0:
            return 0.0
        return max(0.0, 1.0 - self.spent_tokens / self.token_limit)


class AdaptiveBudgetPolicy:
    def __init__(self, uncertainty_threshold: float = 0.35, disagreement_threshold: float = 0.30):
        self.uncertainty_threshold = uncertainty_threshold
        self.disagreement_threshold = disagreement_threshold

    def decide(self, state: BudgetState) -> BudgetAction:
        if state.spent_tokens >= state.token_limit or state.rounds >= state.round_limit:
            return BudgetAction.STOP
        if state.uncertainty < self.uncertainty_threshold and state.disagreement < self.disagreement_threshold:
            return BudgetAction.SYNTHESIZE
        if state.remaining_fraction < 0.25:
            return BudgetAction.CRITIQUE
        if state.disagreement >= self.disagreement_threshold:
            return BudgetAction.CRITIQUE
        return BudgetAction.ADD_AGENT
