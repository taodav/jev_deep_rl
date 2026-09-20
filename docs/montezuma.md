# Montezuma's Revenge prototype

Development is currently paused. The descriptor is available for local experiments;
its platforming and traversal semantics remain incomplete.

`jev_rl/montezuma.py` provides one RAM-to-JSON schema for all rooms. The current
room selects geometry data in `jev_rl/montezuma_rooms.py`; the goal, field meanings,
action criteria, and shared Jev policy do not change room by room. This is an
experimental descriptor, not a solved exploration policy or a safe route planner.

- `room`: zero-based level and room IDs, lighting, and geometry coverage.
- `player`: center position, sprite size, facing, ladder flag, and measured motion.
- `inventory`: keys, swords, torch, and amulet.
- `items`, `hazards`, `doors`: room-local pickups, enemies, beams, and locked doors.
- `geometry`: platforms, walls, ladders, ropes, conveyors, temporary platforms,
  and candidate exits. Only the current room's geometry is sent.
- `local_geometry`: foot height, surfaces aligned with the player's feet, distances
  to their edges, and ladder alignment. These are geometric aids, not action masks.
- `exploration`: distinct rooms observed on this level. Exit destinations are
  remembered only after an observed boundary crossing; unexplored destinations
  stay null. This bounded factual memory resets between episodes; no previous
  input history or room-specific strategy is appended.

All coordinates are **pixels within the current room**, right/down positive.
Actors use center `x/y` and `width/height`; geometry has explicit edge bounds.
Rectangle right/bottom bounds are exclusive; platform tops are standing foot
heights. This is a platform game, so the Ms. Pac-Man cell lattice is not used.
Room changes, life changes, calibrated death animations, and large teleports
invalidate measured motion. Null is unknown, never zero.

The prototype includes 24 reference room templates adapted from
[OCAtari](https://github.com/k4ntz/OC_Atari/blob/master/ocatari/ram/montezumarevenge.py)
under its MIT license (see [third-party notices](../THIRD_PARTY_NOTICES.md)). Tests initialize each room
through a ROM boundary transition after controlled RAM repositioning. Geometry
overlap is checked against rendered pixels in 23 rooms; room 15's uniform
background prevents that check and its template is explicitly unverified.
These calibration transitions are **not policy exploration or gameplay results**.
Starting-room walking, climbing, jumping, death/reset, and a room transition have
also been checked through emulator actions. This does not validate every physics
interaction, higher-level variation, or object state.

Current limitations: sprite/geometry bounds are approximate, object coverage is
partial, and rope attachment, conveyor direction, enemy patrol limits, full
animation/control state, safe drop heights, and jump reachability are not decoded.
Temporary-platform/beam flags use reference RAM definitions and do not predict
future timing. Known templates can expose geometry in dark rooms; this is a RAM
experiment, not a visual-observation benchmark. Dark-room pickups may be unknown.
Exit candidates may be blocked; their existence does not certify traversability.

Local smoke run (no API or credential loading):

```bash
uv run --no-env-file --extra atari --extra recording main.py \
  --env ALE/MontezumaRevenge-v5 --policy random --max-steps 500 \
  --log runs/montezuma-random-500.jsonl --save-npy runs/montezuma-random-500.npy
uv run --no-env-file --extra recording npy_to_gif.py runs/montezuma-random-500.npy --scale 2
```

When a live trial is wanted, the existing bounded wrapper accepts
`--game montezuma`. Its default remains 64 decisions; explicitly set the desired
budget. It saves all requests/responses, RAM, frames, state, and an automatic GIF,
and stops at episode termination. Adding the descriptor does not make API calls.
The first integration checks use the SDK's mock transport.

