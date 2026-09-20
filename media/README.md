# Atari demonstration

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
