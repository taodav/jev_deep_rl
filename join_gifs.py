"""Place animated GIFs side by side, preserving timing until the shortest ends."""

import argparse
import json
from contextlib import ExitStack
from pathlib import Path

from PIL import Image, ImageSequence


def inspect_gif(path: Path) -> dict:
    with Image.open(path) as source:
        if source.format != "GIF":
            raise ValueError("Each input must be a GIF.")
        durations = [int(frame.info.get("duration", 0)) for frame in ImageSequence.Iterator(source)]
        if not durations or any(duration <= 0 for duration in durations):
            raise ValueError("Every source frame needs a positive duration.")
        return {
            "path": str(path),
            "width": source.width,
            "height": source.height,
            "frames": len(durations),
            "duration_ms": sum(durations),
            "frame_durations_ms": durations,
        }


def join_gifs(paths: list[Path], destination: Path, *, crop_bottom: int = 0) -> dict:
    if len(paths) < 2 or destination.suffix.lower() != ".gif":
        raise ValueError("Supply at least two GIFs and an output ending in .gif.")
    details = [inspect_gif(path) for path in paths]
    stop_ms = min(item["duration_ms"] for item in details)
    width = sum(item["width"] for item in details)
    height = max(item["height"] for item in details)
    if crop_bottom < 0 or crop_bottom >= min(item["height"] for item in details):
        raise ValueError("Bottom crop must be nonnegative and smaller than every input height.")
    height -= crop_bottom
    destination.parent.mkdir(parents=True, exist_ok=True)
    # Existing originals and exports are never overwritten.
    stream = destination.open("xb")
    try:
        with stream, ExitStack() as stack:
            sources = [stack.enter_context(Image.open(path)) for path in paths]
            indices = [0] * len(sources)
            ends = [item["frame_durations_ms"][0] for item in details]
            frames, durations = [], []
            elapsed = 0
            while elapsed < stop_ms:
                boundary = min(stop_ms, *ends)
                canvas = Image.new("RGB", (width, height), "black")
                x = 0
                for source in sources:
                    # Pillow resolves GIF disposal and delta frames during seek.
                    canvas.paste(source.convert("RGB"), (x, 0))
                    x += source.width
                frames.append(canvas.quantize(colors=256))
                durations.append(boundary - elapsed)
                elapsed = boundary
                if elapsed == stop_ms:
                    break
                for i, source in enumerate(sources):
                    if ends[i] == elapsed:
                        indices[i] += 1
                        source.seek(indices[i])
                        ends[i] += details[i]["frame_durations_ms"][indices[i]]
            frames[0].save(
                stream,
                format="GIF",
                save_all=True,
                append_images=frames[1:],
                duration=durations,
                loop=0,
                disposal=2,
                optimize=False,
            )
    except BaseException:
        destination.unlink(missing_ok=True)
        raise
    return {
        "output": str(destination),
        "width": width,
        "height": height,
        "duration_s": stop_ms / 1000,
        "composed_frames": len(frames),
        "crop_bottom": crop_bottom,
        "sources_left_to_right": [
            {key: value for key, value in item.items() if key != "frame_durations_ms"}
            for item in details
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("gifs", nargs="+", type=Path, help="inputs in left-to-right order")
    parser.add_argument("--output", required=True, type=Path, help="new output GIF")
    parser.add_argument(
        "--crop-bottom",
        type=int,
        default=0,
        help="pixels to trim from the bottom of the combined canvas",
    )
    args = parser.parse_args()
    print(json.dumps(join_gifs(args.gifs, args.output, crop_bottom=args.crop_bottom), indent=2))


if __name__ == "__main__":
    main()
