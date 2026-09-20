"""Convert trusted Jev harness .npy recordings into looping GIFs, one per episode."""

import argparse
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image


def frame_durations(episode: dict, metadata: dict, fps: float | None) -> list[int]:
    """Use simulation time, rounding cumulative time to GIF's 10 ms resolution."""
    count = len(episode["frames"])
    if fps is not None:
        intervals = np.full(count, 1 / fps)
    elif metadata.get("environment", "").startswith("ALE/"):
        # The harness uses standard NTSC Atari games. ALE's generic render_fps
        # metadata does not account for frame skip; use emulator frame counters.
        nominal = float(metadata.get("frameskip") or 4) / 60
        infos = episode.get("infos", [])
        if len(infos) == count and all("episode_frame_number" in info for info in infos):
            numbers = np.array([info["episode_frame_number"] for info in infos], dtype=float)
            intervals = np.append(np.diff(numbers) / 60, nominal)
        else:
            intervals = np.full(count, nominal)
    else:
        render_fps = metadata.get("environment_metadata", {}).get("render_fps")
        if render_fps is None:
            raise ValueError("No playback timing in recording; supply --fps.")
        intervals = np.full(count, 1 / float(render_fps))
    if not np.all(np.isfinite(intervals)) or np.any(intervals <= 0):
        raise ValueError("Frame timing must be finite and positive.")
    boundaries = np.rint(np.concatenate(([0.0], np.cumsum(intervals))) * 100).astype(np.int64)
    durations = np.diff(boundaries) * 10
    if np.any(durations < 10) or np.any(durations > 655350):
        raise ValueError(
            "Frame timing exceeds GIF's supported range; supply --fps between 1 and 100."
        )
    return durations.tolist()


def convert_recording(
    source: Path,
    *,
    output_dir: Path | None = None,
    fps: float | None = None,
    scale: int = 1,
) -> list[dict]:
    if source.suffix.lower() != ".npy":
        raise ValueError("Input must be a .npy recording.")
    if scale < 1 or (fps is not None and (not math.isfinite(fps) or not 0 < fps <= 100)):
        raise ValueError("Scale must be positive and FPS must be greater than 0 and at most 100.")
    # The harness stores a pickled dictionary. Only load your own trusted runs.
    with source.open("rb") as stream:
        run = np.load(stream, allow_pickle=True).item()
    if (
        not isinstance(run, dict)
        or run.get("format") != "jev-rl-recording"
        or run.get("schema_version") != 1
    ):
        raise ValueError("Expected a Jev harness recording with schema_version 1.")
    episodes = run["episodes"]
    if not episodes:
        raise ValueError("Recording contains no episodes.")
    results = []
    for index, episode in enumerate(episodes):
        frames = episode["frames"]
        if (
            not isinstance(frames, np.ndarray)
            or frames.dtype != np.uint8
            or frames.ndim != 4
            or frames.shape[-1] != 3
            or min(frames.shape) < 1
        ):
            raise ValueError(
                "Each episode must contain nonempty RGB uint8 frames shaped (N, H, W, 3)."
            )
        durations = frame_durations(episode, run["metadata"], fps)
        suffix = f"-episode-{index}" if len(episodes) > 1 else ""
        destination = (output_dir or source.parent) / f"{source.stem}{suffix}.gif"
        destination.parent.mkdir(parents=True, exist_ok=True)
        # Reserve before encoding so existing exports are never overwritten.
        stream = destination.open("xb")
        try:
            with stream:
                images = []
                for frame in frames:
                    image = Image.fromarray(frame)
                    if scale != 1:
                        image = image.resize(
                            (image.width * scale, image.height * scale),
                            Image.Resampling.NEAREST,
                        )
                    images.append(image)
                images[0].save(
                    stream,
                    format="GIF",
                    save_all=True,
                    append_images=images[1:],
                    duration=durations,
                    loop=0,
                    disposal=2,
                    optimize=False,
                )
        except BaseException:
            destination.unlink(missing_ok=True)
            raise
        results.append(
            {
                "output": str(destination),
                "episode": index,
                "source_frames": len(frames),
                "width": frames.shape[2] * scale,
                "height": frames.shape[1] * scale,
                "duration_s": sum(durations) / 1000,
            }
        )
    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("recordings", nargs="+", type=Path, help="trusted harness .npy files")
    parser.add_argument("--output-dir", type=Path, help="default: beside each input file")
    parser.add_argument(
        "--fps", type=float, help="override playback with constant frames per second"
    )
    parser.add_argument(
        "--scale", type=int, default=1, help="integer pixel enlargement (default: 1)"
    )
    args = parser.parse_args(argv)
    if args.scale < 1:
        parser.error("--scale must be a positive integer")
    if args.fps is not None and (not math.isfinite(args.fps) or not 0 < args.fps <= 100):
        parser.error("--fps must be finite, greater than 0 and at most 100")
    try:
        for source in args.recordings:
            for result in convert_recording(
                source, output_dir=args.output_dir, fps=args.fps, scale=args.scale
            ):
                print(json.dumps(result))
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.exit(1, f"GIF conversion failed: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
