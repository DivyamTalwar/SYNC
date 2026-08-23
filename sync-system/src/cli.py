"""Small, fail-closed command line entry point for SYNC."""

from __future__ import annotations

import argparse
import os
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
    args = parser.parse_args(argv)
    if args.command == "doctor":
        return _doctor()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
