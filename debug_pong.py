"""Replay a trusted Pong recording and validate saved Jev decisions without API calls."""

import argparse
import json
import logging
from collections import Counter
from pathlib import Path

import numpy as np

logging.disable(logging.CRITICAL)


def inspect_recording(path: Path, episode_index: int = 0) -> dict:
    # This is the harness's object/pickle format; only inspect trusted recordings.
    run = np.load(path, allow_pickle=True).item()
    if run.get("format") != "jev-rl-recording" or run.get("schema_version") != 1:
        raise ValueError("Expected a harness recording with schema_version 1.")
    metadata = run["metadata"]
    if metadata["environment"] != "ALE/Pong-v5":
        raise ValueError("This debugger currently supports ALE/Pong-v5 recordings.")
    if not 0 <= episode_index < len(run["episodes"]):
        raise ValueError("Episode index is outside the recording.")
    episode = run["episodes"][episode_index]
    count = len(episode["actions"])
    if any(len(episode[field]) != count + 1 for field in ("frames", "observations", "states")):
        raise ValueError("Expected N actions connecting N+1 snapshots.")
    from jev_rl.adapters import Action, make_environment

    checks = {}

    def check(name: str, matches: bool, step: int) -> None:
        item = checks.setdefault(name, {"checked": 0, "mismatches": 0, "first_mismatch_step": None})
        item["checked"] += 1
        if not matches:
            item["mismatches"] += 1
            if item["first_mismatch_step"] is None:
                item["first_mismatch_step"] = step

    # Reproduce the saved experiment, including its old frame skip and sticky seed.
    env, adapter = make_environment(
        "ALE/Pong-v5",
        frameskip=metadata["frameskip"],
        sticky_probability=metadata["sticky_probability"],
        pong_state=metadata["observation_source"],
        render_mode="rgb_array",
    )
    try:
        observation, _ = env.reset(seed=episode["seed"])
        adapter.reset()
        reward = 0.0
        for step in range(count + 1):
            check(
                "raw_observations", np.array_equal(observation, episode["observations"][step]), step
            )
            check("rendered_frames", np.array_equal(env.render(), episode["frames"][step]), step)
            decoded = adapter.encode(observation, reward=reward)
            saved = episode["states"][step]["observation"]
            fields = (
                "player",
                "opponent",
                "ball",
                "ball_motion",
                "ball_relative_to_player",
                "score",
            )
            check(
                "decoded_geometry", all(decoded.get(key) == saved.get(key) for key in fields), step
            )
            if step == count:
                break
            observation, reward, terminated, truncated, _ = env.step(int(episode["actions"][step]))
            check(
                "rewards_and_endings",
                (
                    reward == episode["rewards"][step]
                    and terminated == episode["terminated"][step]
                    and truncated == episode["truncated"][step]
                ),
                step,
            )
    finally:
        env.close()

    validation = {"checked": 0, "rejected": []}
    if metadata["policy"] == "jev":
        import httpx2
        from typesafe_sdk import RetryPolicy, TypeSafeClient, TypeSafeError

        from jev_rl.policies import JevPolicy, PolicyError

        # This transport returns ONLY a saved decision; it cannot access the network.
        current = {}

        def respond(request):
            decision = current["decision"]
            return httpx2.Response(
                200,
                json={
                    "model": decision["model"],
                    "usage": decision["usage"],
                    "answers": {
                        "action": {
                            "type": "choice",
                            "choice": decision["action"],
                            "probabilities": decision["probabilities"],
                            "confidence": decision["confidence"],
                        }
                    },
                },
            )

        actions = tuple(Action(**action) for action in metadata["actions"])
        with TypeSafeClient(
            api_key="offline-placeholder",
            transport=httpx2.MockTransport(respond),
            retry=RetryPolicy(max_retries=0),
        ) as client:
            policy = JevPolicy(client)
            for step, decision in enumerate(episode["decisions"]):
                current["decision"] = decision
                validation["checked"] += 1
                try:
                    policy.decide(episode["states"][step], actions)
                except PolicyError as error:
                    validation["rejected"].append({"step": step, "diagnostics": error.diagnostics})
                except TypeSafeError:
                    validation["rejected"].append(
                        {"step": step, "reason": "sdk_rejected_saved_fields"}
                    )

    observations = [state["observation"] for state in episode["states"][:count]]
    legal_names = {action["name"] for action in metadata["actions"]}
    return {
        "api_requests": 0,
        "episode": episode_index,
        "recording_status": run["status"],
        "original_error_type": run.get("error_type"),
        "original_error_details": run.get("error_details"),
        "frameskip_replayed": metadata["frameskip"],
        "sticky_probability_replayed": metadata["sticky_probability"],
        "decisions_replayed": count,
        "replay_checks": checks,
        "saved_response_validation": validation,
        "decisions_without_ball": sum(state["ball"] is None for state in observations),
        "decisions_with_ball_moving_toward_player": sum(
            state["ball_motion"] is not None and state["ball_motion"]["dx_pixels_per_env_step"] > 0
            for state in observations
        ),
        "action_counts": dict(
            Counter(
                decision["action"] if decision["action"] in legal_names else "invalid"
                for decision in episode["decisions"]
            )
        ),
        "last_saved_state_step": episode["states"][-1]["step"],
        "limitation": "Saved accepted decisions can be replayed; an unsaved rejected response cannot be reconstructed.",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("recording", type=Path)
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--report", type=Path, help="write a new JSON report; never overwrite")
    args = parser.parse_args()
    if args.report is not None and args.report.exists():
        parser.error("report already exists; choose a new filename")
    report = inspect_recording(args.recording, args.episode)
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        with args.report.open("x") as stream:
            json.dump(report, stream, indent=2, allow_nan=False)
            stream.write("\n")
    print(json.dumps(report, indent=2, allow_nan=False))
    return int(
        any(item["mismatches"] for item in report["replay_checks"].values())
        or bool(report["saved_response_validation"]["rejected"])
    )


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        print(
            "Offline inspection failed. Check the recording, episode index, dependencies, and output path."
        )
        raise SystemExit(1) from None
