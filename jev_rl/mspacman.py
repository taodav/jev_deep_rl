"""RAM-only Ms. Pac-Man abstraction for ALE's standard game mode.

Position, pellet-bit and BCD-score register reference:
https://github.com/k4ntz/OC_Atari/blob/master/ocatari/ram/mspacman.py
Life/terminal register reference:
https://github.com/Farama-Foundation/Arcade-Learning-Environment/blob/master/src/ale/games/supported/MsPacman.cpp

Additional registers verified with controlled emulator probes: RAM[1:5] bit 7
is each ghost's frightened flag; RAM[116] low six bits count down once per eight
active frames (high bits change when ghosts are eaten); RAM[39] == 255 denotes
active play. Freeze/restart/scoring animations are deliberately grouped together.
Returning-eye behavior and future ghost routes are not decoded or predicted.
"""

from typing import Any

import gymnasium as gym
import numpy as np

from .adapters import DEFAULT_ATARI_FRAMESKIP, Action
from .mspacman_maze import DIRECTIONS, Y, cell_number, maze_for, maze_position

GHOST_NAMES = ("orange", "cyan", "pink", "red")
# Power pellets occupy the outer vertical corridors, near these lattice cells.
POWER_CELLS = ((32, (1, 0)), (8, (1, 17)), (16, (12, 0)), (4, (12, 17)))


def pellet_register(row: int, col: int) -> tuple[int, int]:
    base = 59 + 3 * row
    if col == 0:
        return base, 6
    if col == 17:
        return base, 4
    if col <= 4:
        return base + 1, 2 * (4 - col) + 1
    if col <= 8:
        return base + 2, 2 * (col - 5)
    if col <= 12:
        return base + 1, 2 * (12 - col)
    return base + 2, 2 * (col - 13) + 1


class MsPacmanRAMAdapter:
    goal = (
        "Clear the maze's pellets and power pellets while preserving lives and "
        "scoring points. Plan routes with escape options. Avoid ghosts unless "
        "individually frightened; frightened ghosts are edible until the power "
        "effect expires. Fruit is an optional bonus."
    )

    def __init__(self, *, frameskip: int = DEFAULT_ATARI_FRAMESKIP):
        self.frameskip = frameskip
        self.reset()

    def reset(self) -> None:
        self._previous: dict | None = None

    def actions(self, env: gym.Env) -> tuple[Action, ...]:
        meanings = env.unwrapped.get_action_meanings()
        descriptions = {
            "NOOP": "Release the joystick.",
            "UP": "Request up.",
            "RIGHT": "Request right.",
            "LEFT": "Request left.",
            "DOWN": "Request down.",
            "UPRIGHT": "Hold up and right.",
            "UPLEFT": "Hold up and left.",
            "DOWNRIGHT": "Hold down and right.",
            "DOWNLEFT": "Hold down and left.",
        }
        if not isinstance(env.action_space, gym.spaces.Discrete) or (
            env.action_space.start != 0
            or env.action_space.n != 9
            or set(meanings) != set(descriptions)
        ):
            raise ValueError("Ms. Pac-Man requires ALE's nine-action minimal action set.")
        return tuple(Action(name.lower(), i, descriptions[name]) for i, name in enumerate(meanings))

    def encode(self, observation: Any, *, reward: float = 0.0) -> dict:
        ram = np.asarray(observation)
        if ram.shape != (128,) or ram.dtype != np.uint8:
            raise ValueError("Ms. Pac-Man needs a 128-byte uint8 RAM observation.")
        layout = int(ram[0])
        maze = maze_for(layout)
        if ram[19] > 3:
            raise ValueError("Unrecognized Ms. Pac-Man ghost count.")
        terminal = int(ram[123]) & 15 == 0 and ram[39] == 83
        phase = "game_over" if terminal else "active" if ram[39] == 255 else "animation_or_restart"
        lives = 0 if terminal else (int(ram[123]) & 7) + 1
        epoch = (layout, lives, phase)
        previous = (
            self._previous
            if self._previous and self._previous["epoch"] == epoch and phase == "active"
            else None
        )
        positions = {}

        def actor(name: str, x_register: int, y_register: int) -> dict:
            x, y = int(ram[x_register]) - 9, int(ram[y_register]) + 5.5
            positions[name] = (x, y)
            row, column = maze_position(x, y)
            motion = None
            if previous and name in previous["positions"]:
                old_x, old_y = previous["positions"][name]
                dx, dy = x - old_x, y - old_y
                in_tunnel = abs(dy) < 2 and any(abs(y - Y[r]) < 2 for r in maze.tunnel_rows)
                if in_tunnel and abs(dx) > 100:
                    dx += -160 if dx > 0 else 160
                if abs(dx) + abs(dy) <= max(8, 3 * self.frameskip):
                    old_row, old_column = maze_position(old_x, old_y)
                    dcolumn = column - old_column
                    # Fractional columns also wrap when passing cell 0, which
                    # occurs at a different point than the RAM coordinate wrap.
                    if in_tunnel and abs(dcolumn) > 9:
                        dcolumn += -18 if dcolumn > 0 else 18
                    motion = {"rows": cell_number(row - old_row), "columns": cell_number(dcolumn)}
            return {
                "row": cell_number(row),
                "column": cell_number(column),
                "motion_per_decision": motion,
            }

        player = actor("player", 10, 16)
        player["heading"] = ("up", "right", "down", "left")[int(ram[56]) & 3]
        location = maze.locate(*positions["player"])
        # Use the current corridor segment between cells; rounding to a nearby
        # junction would advertise turns before the player reaches the opening.
        player.update(
            {
                f"wall_{direction}": None
                if location is None
                else direction not in location["links"]
                for direction, _, _ in DIRECTIONS
            }
        )
        distances = maze.distances_in_cells(location)
        ghosts = []
        for i, name in enumerate(GHOST_NAMES[: int(ram[19]) + 1]):
            ghost = actor(name, 6 + i, 12 + i)
            ghost_x, ghost_y = positions[name]
            ghost.update(
                {
                    "id": name,
                    "frightened": bool(int(ram[i + 1]) & 128),
                    "in_house": 64 < ghost_x < 96 and 64 < ghost_y < 96,
                }
            )
            ghosts.append(ghost)

        pellets = set()
        for row, col in maze.graph:
            register, bit = pellet_register(row, col)
            if int(ram[register]) & (1 << bit):
                pellets.add((row, col))
        powers = [cell for mask, cell in POWER_CELLS if int(ram[117]) & mask]
        rows = [list(row.replace(".", " ")) for row in maze.rows]
        for r, c in pellets:
            rows[r][c] = "."
        for r, c in powers:
            rows[r][c] = "o"
        fruit = actor("fruit", 11, 17) if ram[11] > 0 and ram[17] > 0 else None
        self._previous = {"epoch": epoch, "positions": positions}
        score = sum(
            ((int(ram[120 + i]) >> 4) * 10 + (int(ram[120 + i]) & 15)) * 100**i for i in range(3)
        )
        ticks = int(ram[116]) & 63
        state = {
            "emulator_frames_per_action": self.frameskip,
            "phase": phase,
            "score": score,
            "lives_remaining": lives,
            "player": player,
            "ghosts": ghosts,
            "fruit": fruit,
            "power_effect": {"remaining_active_frames_upper_bound": ticks * 8},
            "maze": {
                "rows_top_to_bottom": ["".join(row) for row in rows],
                "tunnel_rows": list(maze.tunnel_rows),
                "pellets_remaining": len(pellets),
                "legend": "# wall/ghost house; space empty corridor; . pellet; o power pellet (approximate cell). Orthogonal non-wall neighbors connect. tunnel_rows link columns 0 and 17. Actors are listed separately.",
            },
            "navigation": {
                "corridors_from_player": maze.corridors(location, pellets),
                "nearest_pellets": [
                    {
                        "row": cell[0],
                        "column": cell[1],
                        "distance_cells": cell_number(distances[cell]),
                    }
                    for cell in sorted(
                        pellets & distances.keys(), key=lambda cell: (distances[cell], cell)
                    )[:3]
                ],
            },
        }
        state["description"] = (
            "All positions use zero-based maze row and column; fractions lie between cell centers. "
            "Rows increase down, columns right. Columns 17 through 18 cross a tunnel to column 0. "
            "Motion is row/column displacement per decision, distinct from heading. "
            "wall_* is true where maze geometry blocks that direction at your current position, "
            "false for an opening (including tunnels), null if unknown; it ignores ghosts. "
            "Distances count cell connections, not time. Corridors lead to the next corner/junction "
            "and do not predict ghost movement. "
            "Turns require an opening; a blocked turn or noop can continue movement. "
            "Diagonals request two cardinal directions. Returning-eye status is unknown. "
            "The power countdown estimates remaining active frames; animations pause it. "
            "animation_or_restart temporarily prevents control."
        )
        return state
