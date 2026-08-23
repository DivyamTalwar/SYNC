"""Small, fail-closed command line entry point for SYNC."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from importlib.metadata import version


def _doctor() -> int:
    checks = {
        "OPENROUTER_API_KEY": bool(os.getenv("OPENROUTER_API_KEY")),
        "COHERE_API_KEY": bool(os.getenv("COHERE_API_KEY")),
    }
    print(f"SYNC {version('sync-system')} environment")
    for name, configured in checks.items():
        print(f"- {name}: {'configured' if configured else 'missing'}")
    if not all(checks.values()):
        print("Provider-backed collaboration is unavailable until every required key is configured.")
        return 1
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="sync", description="SYNC research runtime utilities")
    subcommands = parser.add_subparsers(dest="command", required=True)
    subcommands.add_parser("doctor", help="validate provider configuration without making API calls")
    trace_parser = subcommands.add_parser("trace-verify", help="verify a hash-chained cognitive trace")
    trace_parser.add_argument("path", type=Path)
    args = parser.parse_args(argv)
    if args.command == "doctor":
        return _doctor()
    if args.command == "trace-verify":
        from src.observability.trace import CognitiveTrace

        valid = CognitiveTrace(args.path).verify()
        print("valid" if valid else "invalid")
        return 0 if valid else 1
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
