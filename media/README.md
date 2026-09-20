# Atari demonstrations

## Policy showcase

[`jev-atari-showcase-no-header.gif`](jev-atari-showcase-no-header.gif) presents all three games
with recorded action probabilities, animated joystick inputs, scores, and decision
numbers. All three runs use one shared model and action-question template;
state adapters, goals, and action definitions are game-specific.

[Static preview](jev-atari-showcase-no-header.png) · [Render manifest](jev-atari-showcase-no-header.json)

The default layout omits the title and subtitle, keeping the game cards and
minimal time footer. The 1200 × 768 animation plays at 2× game speed for 30.96 seconds and loops.
It samples the source recordings at 15 playback frames per second. API waiting
time is omitted. The Atari-inspired console styling is drawn in code; no external
logo images or font files are bundled.

For each displayed frame, the probabilities and joystick show the decision made
from that frame's state. They indicate the model's requested action; ALE sticky
actions can repeat a previous emulator input. Probability bars retain all legal
options in a fixed order, including fire combinations and diagonal directions.
At terminal states there is no subsequent decision. The three runs play alongside
one another using emulator time and stop when the shortest recording finishes.

Reproduce it from the trusted local recordings (choose a new output name):

```bash
uv run --locked --no-env-file --extra recording render_showcase.py \
  runs/jev-2000-fs2/pong-ram-jev-trial-2000.npy \
  runs/jev-10000-fs2-breakout-mspacman/breakout-ram-jev-trial-10000.npy \
  runs/jev-10000-fs2-breakout-mspacman/mspacman-ram-jev-trial-10000.npy \
  --output runs/showcase-v2/jev-atari.gif
```

The renderer creates a GIF, PNG preview, and JSON manifest without any API calls.
It uses the first episode from each input, sorts games into Pong / Breakout /
Ms. Pac-Man order, and checks that the model and question instructions match.
Use `--duration 4` for a short layout iteration, `--speed` and `--fps` to adjust
playback, and `--preview-at` to select the PNG's time in playback seconds.
Use `--with-header` to include the optional title and subtitle.
Existing outputs are never overwritten. Typography uses installed system fonts
with a Pillow fallback, so exact text rendering can differ between systems.

## Compact gameplay GIF

`jev-atari-2x.gif` shows Jev playing Pong, Breakout, and Ms. Pac-Man from left to
right. Each policy uses RAM observations and two emulator frames per decision.
Playback runs at 2× game speed and excludes time spent waiting for API responses.

The combined recording stops at the shortest source duration. It includes 61.92
seconds of game time, played over 30.96 seconds, and loops continuously. Blank
bottom margins were cropped.

This copy halves the existing 2× GIF's width and height using nearest-neighbor
resampling, from 960 × 392 to 480 × 196 pixels. Frame timing is preserved.

Source recordings, retained locally under the ignored `runs/` directory:

- Pong: `jev-2000-fs2/pong-ram-jev-trial-2000.gif`
- Breakout: `jev-10000-fs2-breakout-mspacman/breakout-ram-jev-trial-10000.gif`
- Ms. Pac-Man: `jev-10000-fs2-breakout-mspacman/mspacman-ram-jev-trial-10000.gif`

The immediate source is
`runs/jev-10000-fs2-breakout-mspacman/pong-breakout-mspacman-cropped-2x-downsampled.gif`.
The demonstration uses selected clips from individual runs.
