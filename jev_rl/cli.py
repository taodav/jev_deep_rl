"""Run local baseline episodes or explicit, bounded Jev evaluation episodes."""

import argparse
import json
import logging
import sys
from contextlib import ExitStack
from dataclasses import asdict
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from uuid import uuid4

from typesafe_sdk import RetryPolicy, TypeSafeClient, TypeSafeError

from .adapters import DEFAULT_ATARI_FRAMESKIP, make_environment
from .policies import JevPolicy, PolicyError, RandomPolicy, action_question
from .recording import NpyRecorder
from .responses import CaptureTransport, ReplayThenLiveTransport, ResponseCapture
from .runner import DEFAULT_MAX_STEPS, run_episode, write_event


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be a nonnegative integer")
    return parsed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--env",
        choices=(
            "CartPole-v1",
            "ALE/Pong-v5",
            "ALE/Breakout-v5",
            "ALE/MsPacman-v5",
            "ALE/MontezumaRevenge-v5",
        ),
        default="CartPole-v1",
    )
    parser.add_argument("--policy", choices=("random", "jev"), default="random")
    parser.add_argument("--episodes", type=positive_int, default=1)
    parser.add_argument(
        "--max-steps",
        type=positive_int,
        default=DEFAULT_MAX_STEPS,
        help="decision budget per episode (one environment step per decision)",
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--model", default="jev-1.13.0", help="pinned model by default; jev-latest is also accepted"
    )
    parser.add_argument(
        "--max-retries",
        type=nonnegative_int,
        default=2,
        help="HTTP retries per Jev decision; use 0 for a strict request cap",
    )
    parser.add_argument(
        "--frameskip",
        type=positive_int,
        default=DEFAULT_ATARI_FRAMESKIP,
        help="emulator frames per Atari environment step",
    )
    parser.add_argument(
        "--pong-state",
        choices=("rgb", "ram"),
        default="rgb",
        help="source for Pong's structured observation adapter",
    )
    parser.add_argument("--sticky-probability", type=float, default=0.25)
    parser.add_argument(
        "--log", type=Path, help="new JSONL output file; existing files are never overwritten"
    )
    parser.add_argument(
        "--responses",
        type=Path,
        help="Jev response journal; defaults to NAME.responses.jsonl beside the run output",
    )
    parser.add_argument(
        "--replay-responses",
        type=Path,
        help="reuse a saved response prefix, then call Jev for remaining steps; requires one episode and no retries",
    )
    parser.add_argument(
        "--save-npy", type=Path, help="save RGB frames and all rollout data to a new .npy file"
    )
    args = parser.parse_args(argv)
    if args.seed < 0:
        parser.error("--seed must be nonnegative")
    if not 0 <= args.sticky_probability <= 1:
        parser.error("--sticky-probability must be between 0 and 1")
    if args.save_npy is not None and args.save_npy.suffix.lower() != ".npy":
        parser.error("--save-npy must end in .npy")
    if args.responses is not None and (
        args.policy != "jev" or args.responses.suffix.lower() != ".jsonl"
    ):
        parser.error("--responses requires --policy jev and a .jsonl filename")
    if args.replay_responses is not None and (
        args.policy != "jev" or args.episodes != 1 or args.max_retries != 0
    ):
        parser.error("--replay-responses requires --policy jev, --episodes 1, and --max-retries 0")
    # No dotenv import or file loading. The SDK uses the process environment.
    try:
        with ExitStack() as stack:
            recorder = (
                stack.enter_context(NpyRecorder(args.save_npy))
                if args.save_npy is not None
                else None
            )
            env, adapter = make_environment(
                args.env,
                frameskip=args.frameskip,
                sticky_probability=args.sticky_probability,
                render_mode="rgb_array" if recorder is not None else None,
                pong_state=args.pong_state,
            )
            stack.callback(env.close)
            log = None
            if args.log is not None:
                args.log.parent.mkdir(parents=True, exist_ok=True)
                log = stack.enter_context(args.log.open("x", encoding="utf-8"))
            if args.policy == "jev":
                logging.disable(logging.CRITICAL)
                destination = args.log or args.save_npy
                if args.responses is None:
                    args.responses = (
                        destination.with_name(destination.stem + ".responses.jsonl")
                        if destination is not None
                        else Path("runs")
                        / f"jev-{datetime.now(timezone.utc):%Y%m%dT%H%M%S}-{uuid4().hex[:8]}.responses.jsonl"
                    )
                capture = stack.enter_context(
                    ResponseCapture(
                        args.responses,
                        on_record=recorder.api_responses.append if recorder is not None else None,
                    )
                )
                replay = (
                    ReplayThenLiveTransport(
                        args.replay_responses, seed=args.seed, max_steps=args.max_steps
                    )
                    if args.replay_responses is not None
                    else None
                )
                transport = CaptureTransport(capture, replay)
                stack.callback(transport.close)
                client = stack.enter_context(
                    TypeSafeClient(
                        model=args.model,
                        transport=transport,
                        retry=RetryPolicy(max_retries=args.max_retries, timeout=10.0),
                        timeout=10.0,
                    )
                )
                policy = JevPolicy(client, capture)
            else:
                policy = RandomPolicy()
            actions = adapter.actions(env)
            packages = ["gymnasium", "numpy", "typesafe-sdk"]
            if args.env.startswith("ALE/"):
                packages.append("ale-py")
            if recorder is not None and args.env == "CartPole-v1":
                packages.append("pygame-ce")
            metadata = {
                "event": "run_start",
                "schema_version": 1,
                "environment": args.env,
                "policy": args.policy,
                "observation_source": (
                    "ram"
                    if args.env in {"ALE/Breakout-v5", "ALE/MsPacman-v5", "ALE/MontezumaRevenge-v5"}
                    else args.pong_state
                    if args.env == "ALE/Pong-v5"
                    else "vector"
                ),
                "adapter": type(adapter).__name__,
                "requested_model": args.model if args.policy == "jev" else None,
                "episodes": args.episodes,
                "max_steps": args.max_steps,
                "seed": args.seed,
                "frameskip": args.frameskip if args.env.startswith("ALE/") else None,
                "sticky_probability": args.sticky_probability
                if args.env.startswith("ALE/")
                else None,
                "max_http_attempts_per_decision": 1 + args.max_retries
                if args.policy == "jev"
                else 0,
                "responses_log": str(args.responses) if args.policy == "jev" else None,
                "replay_responses": str(args.replay_responses)
                if args.replay_responses is not None
                else None,
                "versions": {package: version(package) for package in packages},
                "actions": [asdict(action) for action in actions],
                "questions": {"action": action_question(actions).model_dump()},
                "environment_metadata": env.metadata,
            }
            write_event(log, metadata)
            if recorder is not None:
                recorder.metadata = metadata
            for episode in range(args.episodes):
                recording = (
                    recorder.start_episode(episode, args.seed + episode)
                    if recorder is not None
                    else None
                )
                result = run_episode(
                    env,
                    adapter,
                    policy,
                    seed=args.seed + episode,
                    max_steps=args.max_steps,
                    episode=episode,
                    log=log,
                    recording=recording,
                )
                print(json.dumps(asdict(result), allow_nan=False))
    except TypeSafeError:
        # Do not print exceptions or HTTP payloads, which can carry request data.
        print(
            "TypeSafe request failed. Check the API key environment variable, account access, "
            "and connection. See the redacted response journal for saved details.",
            file=sys.stderr,
        )
        return 1
    except ImportError:
        print(
            "Environment dependency missing. Run: uv sync --extra atari --extra recording",
            file=sys.stderr,
        )
        return 1
    except OSError, ValueError, PolicyError:
        print(
            "Run failed: check environment configuration and output destinations "
            "(they must be new files).",
            file=sys.stderr,
        )
        return 1
    return 0
