"""Deterministic batching primitives shared by embedding providers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Generic, Sequence, TypeVar


T = TypeVar("T")


@dataclass(frozen=True)
class DeduplicatedBatch:
    """Unique request values plus an inverse map to original order."""

    unique_texts: tuple[str, ...]
    inverse_indices: tuple[int, ...]

    @classmethod
    def from_texts(cls, texts: Sequence[str]) -> "DeduplicatedBatch":
        if not texts:
            raise ValueError("at least one text is required")

        index_by_text: dict[str, int] = {}
        unique: list[str] = []
        inverse: list[int] = []
        for text in texts:
            if not isinstance(text, str) or not text.strip():
                raise ValueError("embedding texts must be non-empty strings")
            index = index_by_text.get(text)
            if index is None:
                index = len(unique)
                index_by_text[text] = index
                unique.append(text)
            inverse.append(index)
        return cls(tuple(unique), tuple(inverse))

    def restore(self, unique_values: Sequence[T]) -> list[T]:
        if len(unique_values) != len(self.unique_texts):
            raise ValueError("unique value count does not match the batch plan")
        return [unique_values[index] for index in self.inverse_indices]
