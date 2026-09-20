"""Bounded live Jev trial, with no HTTP retries (64 decisions per game by default).

Run with `uv run --env-file .env --extra atari --extra recording run_jev_trial.py`.
Only uv reads the credential file. This script never opens or parses it.
"""

import argparse
import json
import os
import sys
from pathlib import Path

from jev_rl.runtime import configure_runtime

# Apply before importing the SDK; a user's debug setting must not dump requests.
configure_runtime()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--game",
        choices=("both", "pong", "breakout", "mspacman", "montezuma"),
        default="both",
        help="select a game; 'both' continues to mean Pong and Breakout",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("runs"),
        help="use a new directory for another bounded trial",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=64,
        help="maximum decisions per game; stop earlier if the episode ends",
    )
    parser.add_argument(
        "--replay-responses",
        type=Path,
        help="reuse this game's saved replies to finish its decision budget",
    )
    args = parser.parse_args(argv)
    if args.max_steps < 1:
        parser.error("--max-steps must be positive")
    if args.replay_responses is not None and args.game == "both":
        parser.error("--replay-responses requires a single --game")
    if not os.environ.get("TYPESAFE_API_KEY"):
        print("Trial stopped: API credentials are unavailable.", file=sys.stderr)
        return 1
    from jev_rl.cli import main as run_environment
    from npy_to_gif import convert_recording

    games = ("pong", "breakout") if args.game == "both" else (args.game,)
    destinations = [args.output_dir / f"{game}-ram-jev-trial-{args.max_steps}" for game in games]
    # Validate every destination before the first API request.
    if any(
        stem.with_suffix(suffix).exists()
        for stem in destinations
        for suffix in (".jsonl", ".npy", ".gif", ".responses.jsonl")
    ):
        print(
            "Trial stopped: an output already exists; preserve it and choose new destinations.",
            file=sys.stderr,
        )
        return 1
    env_names = {
        "pong": "Pong",
        "breakout": "Breakout",
        "mspacman": "MsPacman",
        "montezuma": "MontezumaRevenge",
    }
    for game, stem in zip((env_names[name] for name in games), destinations):
        print(
            json.dumps({"trial": "started", "game": game, "max_requests": args.max_steps}),
            flush=True,
        )
        status = run_environment(
            [
                "--env",
                f"ALE/{game}-v5",
                "--policy",
                "jev",
                "--pong-state",
                "ram",
                "--episodes",
                "1",
                "--max-steps",
                str(args.max_steps),
                "--max-retries",
                "0",
                "--model",
                "jev-1.13.0",
                "--seed",
                "7",
                "--log",
                str(stem.with_suffix(".jsonl")),
                "--save-npy",
                str(stem.with_suffix(".npy")),
            ]
            + (
                ["--replay-responses", str(args.replay_responses)]
                if args.replay_responses is not None
                else []
            )
        )
        if status:
            # A failed decision can still leave a useful partial recording.
            try:
                for result in convert_recording(stem.with_suffix(".npy"), scale=2):
                    print(json.dumps(result), flush=True)
            except OSError, ValueError, KeyError, TypeError:
                print("No partial GIF could be exported.", file=sys.stderr)
            return status  # Never spend more requests after an error.
        for result in convert_recording(stem.with_suffix(".npy"), scale=2):
            print(json.dumps(result), flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        # Avoid traceback/exception reprs, which may contain request credentials.
        print("Trial stopped unexpectedly. Raw error details are suppressed.", file=sys.stderr)
        raise SystemExit(1) from None
