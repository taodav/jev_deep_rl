# Observation adapters

The Pong and Breakout adapters include an English `observation.description` alongside named
numeric fields. It states which paddle the agent controls, the coordinate origin
and directions, action duration, ball position relative to the paddle, and measured
ball motion. Pong RAM adds scores; Breakout adds launch status, lives, score, and the
brick map's orientation and legend. Missing positions or motion are explicitly
unknown. Motion describes the last step, not a forecast through future collisions.

These summaries are assembled deterministically from the decoded facts. They do
not choose an action or compute an interception target. The Choice question tells
Jev to use the description to interpret the state, and each candidate explains the
corresponding control. Emulator tests validate the inputs; whether Jev uses them
effectively still requires live policy evaluation.

Each request contains the current observation, goal, decision index, previous
action, and previous reward. No previous-input history is appended. Adapters
compute motion from consecutive observations and reset it at game boundaries.
The policy executes Jev's returned Choice label; preprocessing does not select
a control action.

**CartPole:** the four observation values become named cart-position, velocity,
pole-angle and angular-velocity fields. Jev chooses `push_left` or `push_right`.
This is a small check of the integration, not a claim about control performance.

**Pong RGB (`--pong-state rgb`, the default):** a deterministic palette detector extracts the ball and both paddles from
the standard `(210, 160, 3)` RGB observation. It excludes the score display and gray
borders, estimates displacement between successive observations, and supplies
above/below/overlapping relationships. Missing detections are `null`; motion
history resets on disappearance, scoring, and episode reset. No RAM or hidden
emulator state is sent. The detector is specific to standard Pong colors, resolution,
and mode; it does not accept resized/grayscale/stacked frames or other Atari games.

The palette and crop are cross-checked against the primary
[OCAtari Pong detector](https://github.com/k4ntz/OC_Atari/blob/master/ocatari/vision/pong.py).
We use NumPy directly; OCAtari is not a dependency. The geometric preprocessing does
not calculate an interception target or choose an action.

**Pong RAM (`--pong-state ram`):** ALE returns a `(128,)` `uint8` observation instead
of an image. A small game-specific decoder converts memory registers into ball and
paddle coordinates plus the score; the same motion and alignment calculations then
produce structured state for Jev. The policy does not receive the 128 unexplained
bytes or use pixels. With `--save-npy`, the raw RAM observations and separate RGB
renders are both recorded.

| RAM index (decimal) | Meaning |
| --- | --- |
| 49, 54 | Ball horizontal and vertical position |
| 50, 51 | Opponent and player paddle vertical position |
| 13, 14 | Opponent and player score |

The register map and offsets come from the primary
[OCAtari RAM decoder](https://github.com/k4ntz/OC_Atari/blob/master/ocatari/ram/pong.py).
Our adapter converts them to approximate screen coordinates and clips nominal sprite
boxes to the playfield. The RAM/pixel comparison test checks object centers within
one pixel on sampled gameplay; this is not a guarantee of identical render geometry
in every ROM mode. RAM can describe objects even when their sprites are not visible.

For example, use RAM with Jev:

```bash
uv run --env-file .env --extra atari main.py --env ALE/Pong-v5 \
  --pong-state ram --policy jev --save-npy runs/pong-ram-jev.npy
```

RAM observations are built into [ALE's Pong interface](https://ale.farama.org/environments/pong/).
Their meanings are specific to each game's ROM; adding other Atari games requires
another decoder. RAM is a useful way to test decisions without a visual detector,
but represents a different observation setup from a pixel-based Atari benchmark.

The six legal Pong actions are named `stay`, `fire`, `up`, `down`, `up_fire`, and
`down_fire`. The adapter obtains the action indices from ALE. Its joystick labels
`RIGHT` and `LEFT` move the Pong paddle up and down, respectively; an emulator test
verifies this mapping. All six minimal actions are exposed to both policies.

**Breakout RAM:** `ALE/Breakout-v5` uses RAM observations automatically. The decoder
exposes the bottom paddle, ball position and measured motion, relative offsets,
whether a ball is in play, lives remaining, score, and a 6-by-18 brick bitmap with
row/column counts. Actions are `stay`, `launch`, `right`, and `left`; `launch` presses
Fire to serve when the game is ready. See [ALE's Breakout interface](https://ale.farama.org/environments/breakout/).

| RAM index (decimal) | Meaning |
| --- | --- |
| 72 | Paddle horizontal position |
| 99, 101 | Ball horizontal and vertical position |
| 57 | Lives remaining |
| 76, 77 | Score, in packed decimal digits |
| 0–35 | Brick occupancy bits |

The register map follows the primary
[OCAtari Breakout RAM decoder](https://github.com/k4ntz/OC_Atari/blob/master/ocatari/ram/breakout.py).
Tests compare the brick layout and sampled ball positions to rendered gameplay,
and check serving and both paddle directions. Paddle geometry is approximate:
the nominal width is 16 pixels, but the current width and shrinkage are not decoded.
The state explicitly reports the width as unknown rather than assuming a full-width
paddle throughout the game. Life loss clears motion history; a positive brick reward
does not, since the ball can remain in play.

To run Jev on Breakout, use the runtime credential loader:

```bash
uv run --env-file .env --extra atari main.py --env ALE/Breakout-v5 --policy jev \
  --save-npy runs/breakout-ram-jev.npy --log runs/breakout-ram-jev.jsonl
```

The default Atari settings use two emulator frames per decision and
sticky-action probability 0.25. Configure them with `--frameskip` and
`--sticky-probability`. This is a synchronous loop: the simulator waits during an API
call. Network latency reduces wall-clock throughput, not the freshness of the state.
It is not a real-time 60-FPS controller. Displacement is per environment step and
can hide intermediate bounces; frame skipping is an experimental variable.
The 500-decision default remains unchanged: at the default frame skip 2 this covers
up to 1,000 emulator frames (roughly 16.7 seconds at 60 Hz). Frame skip 4 covers
twice as much game time for the same decision budget, or uses half as many decisions
for the same game duration.

**Ms. Pac-Man RAM:** `ALE/MsPacman-v5` uses the same single-Choice Jev policy and
recording pipeline. The game-specific abstractor sends:

- Player and four ghost positions in maze rows/columns, measured displacement, player heading, and
  individual frightened flags; fruit position when present.
- A compact 14-by-18 maze: `#` for walls/ghost house, spaces for cleared corridors,
  `.` for pellets, and `o` for power pellets. Four static layouts are selected by
  RAM; the remaining food comes from RAM bits. Actor positions are separate, so
  they do not obscure food in the text map.
- Wraparound tunnel rows, corridors available from the player's position, the
  distance in cell connections and exits at each next corner/junction, pellets along those segments,
  and the three nearest pellets by maze distance. These are geometric facts, not
  an action chosen by a local controller or a prediction of ghost movement.
- Score, lives including the current life, active/animation/game-over phase, and
  an approximate upper bound on remaining power-effect time in active frames.

All positions use named, zero-based `row` and `column` fields, increasing down and
right respectively. Fractional values preserve positions between cell centers:
`{"row": 8, "column": 8.5}` lies halfway between columns 8 and 9. The bottom-left
corner is `{"row": 13, "column": 0}`. Ghosts inside the house retain their
coordinates and an `in_house` flag, without being snapped to a player corridor.
Tunnel positions traverse columns 17 through 18, where 18 wraps to column 0.

`motion_per_decision` gives `rows` and `columns` of observed displacement, or
`null` when unknown. Heading is separate: `"heading": "left"` with zero motion
means facing left while stationary. Positions and motion are rounded to three
decimal places after calculating displacement, including tunnel wrapping.

The player also has `wall_up`, `wall_right`, `wall_down`, and `wall_left`.
`true` means maze geometry blocks travel in that direction from the current
position; `false` means an opening, including a wraparound tunnel. Between cell
centers these describe the current corridor segment, without rounding ahead to
a junction. If the player cannot be located on a valid corridor, all four are
`null` (unknown). Ghosts, food, animation state, and sticky actions do not change
these geometry flags. At the bottom-left corner, up/right are `false` and
down/left are `true`.

Route `distance_cells` counts adjacent-cell connections, with fractional distance
to the first cell when between cells. One connection counts as one unit even
through the center gap or a tunnel. These distances are neither elapsed time nor
time to collision. The three nearest pellets are ranked by this cell-distance
metric. Adjacent non-wall samples connect through cleared corridors; the ghost
house is not a player route. Power-pellet markers represent approximate cells.

The request has one short goal, one coordinate/control description, and one maze
legend. Score, lives, phase, and action duration appear once as data. Power-pellet
locations appear only in the map. RAM source and layout ID are not repeated
in policy observations; the source remains in run metadata and layout is in raw
RAM. Total pellet count and navigation summaries are retained to spare the model
counting the map and reconstructing local routes. Raw RAM and RGB frames are still
saved alongside the exact policy states and API responses in NPY recordings.

Internally the adapter still decodes pixel centers to preserve ROM geometry:
`y=7.5+12*row`, `x=9+8*column+(4 if column>=9 else 0)` at integer cell centers.
Interpolation accounts for the 12-pixel middle gap and 20-pixel tunnel link;
ordinary column gaps are 8 pixels. These pixel positions and formulas are not
sent to Jev.

The nine actions are `noop`, four cardinal directions, and four diagonal joystick
combinations. `noop` releases the joystick and can continue existing movement;
it is not a stop command. A diagonal requests two directions and the game resolves
cardinal travel. Turns require an opening. No fire action is needed. The default
sticky-action probability also applies, so an action can be repeated by ALE.

The goal asks Jev to clear the maze while preserving lives, avoid unconfirmed
ghosts, use power pellets and escape routes, and treat fruit as an optional bonus.
There is no appended state history or extra model call. Motion uses the last two
RAM observations and resets across lives, layouts, animation boundaries, and
implausible jumps; normal tunnel wrapping is handled explicitly.

Register references are the primary
[OCAtari Ms. Pac-Man decoder](https://github.com/k4ntz/OC_Atari/blob/master/ocatari/ram/mspacman.py)
and [ALE game settings](https://github.com/Farama-Foundation/Arcade-Learning-Environment/blob/master/src/ale/games/supported/MsPacman.cpp).
Maze geometry was measured offline with the installed ALE 0.12.1 ROM, not inferred
from remaining pellets. Runtime policy input never consults pixels. Tests check
all four maps against rendered walls/pellets, controls, power consumption, ghost
edibility/expiry, life/score changes, and the complete mock API/NPY/GIF path.

RAM[1:5] bit 7 supplies each frightened flag. RAM[116]'s low six bits are the
power countdown; its upper bits also change when ghosts are eaten and must not be
included in time remaining. These fields were verified with controlled emulator
probes. Returning eyes and exact ghost behavior are **not decoded**; this limitation
is stated once in the shared description, and the goal treats non-frightened ghosts cautiously.
`animation_or_restart` groups startup, life loss, and scoring/level animations.
The maps target ALE's standard mode/ROM; unknown layout IDs fail explicitly.
This is a RAM-plus-known-maze experiment, not a pixel-only Atari benchmark.


See [Montezuma's Revenge](montezuma.md) for the experimental platforming descriptor.
