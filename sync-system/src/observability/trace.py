"""Append-only, hash-chained cognitive traces."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Iterable


GENESIS_HASH = "0" * 64


def _canonical_json(value: dict[str, Any]) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


@dataclass(frozen=True)
class TraceEvent:
    sequence: int
    event_type: str
    payload: dict[str, Any]
    previous_hash: str
    event_hash: str


class CognitiveTrace:
    """Write and verify deterministic traces without storing provider secrets."""

    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        events = list(self.read()) if self.path.exists() else []
        self._sequence = len(events)
        self._previous_hash = events[-1].event_hash if events else GENESIS_HASH

    def append(self, event_type: str, payload: dict[str, Any]) -> TraceEvent:
        if not event_type.strip():
            raise ValueError("event_type is required")
        body = {
            "sequence": self._sequence,
            "event_type": event_type,
            "payload": payload,
            "previous_hash": self._previous_hash,
        }
        event_hash = sha256(_canonical_json(body).encode()).hexdigest()
        event = TraceEvent(event_hash=event_hash, **body)
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(_canonical_json(asdict(event)) + "\n")
        self._sequence += 1
        self._previous_hash = event_hash
        return event

    def read(self) -> Iterable[TraceEvent]:
        if not self.path.exists():
            return []
        events: list[TraceEvent] = []
        with self.path.open(encoding="utf-8") as stream:
            for line in stream:
                if line.strip():
                    events.append(TraceEvent(**json.loads(line)))
        return events

    def verify(self) -> bool:
        previous = GENESIS_HASH
        expected_sequence = 0
        for event in self.read():
            if event.sequence != expected_sequence or event.previous_hash != previous:
                return False
            body = {
                "sequence": event.sequence,
                "event_type": event.event_type,
                "payload": event.payload,
                "previous_hash": event.previous_hash,
            }
            if sha256(_canonical_json(body).encode()).hexdigest() != event.event_hash:
                return False
            previous = event.event_hash
            expected_sequence += 1
        return True
