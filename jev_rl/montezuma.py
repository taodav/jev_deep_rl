"""Experimental RAM state descriptor shared across Montezuma's rooms.

No policy, route planner, physics simulator, or API calls live here. RAM field
references: OCAtari (see THIRD_PARTY_NOTICES.md) and ALE's MontezumaRevenge.cpp.
Room templates and sprite boxes are approximations; see README for coverage.
"""

from typing import Any

import gymnasium as gym
import numpy as np

from .adapters import DEFAULT_ATARI_FRAMESKIP, Action
from .montezuma_rooms import local_geometry, room_geometry


class MontezumaRAMAdapter:
    goal = (
        "Explore Montezuma's Revenge, collect treasure and useful tools, and progress "
        "through rooms while preserving lives. Keys open locked doors. Use platforms, "
        "ladders and ropes to navigate; avoid enemies, beams, gaps and dangerous falls. "
        "Choose actions toward a reachable useful objective. Geometric proximity alone "
        "does not establish a safe route."
    )

    def __init__(self, *, frameskip: int = DEFAULT_ATARI_FRAMESKIP):
        self.frameskip = frameskip
        self.reset()

    def reset(self):
        self._previous = None
        self._visited = set()
        self._destinations = {}

    def actions(self, env: gym.Env) -> tuple[Action, ...]:
        meanings = env.unwrapped.get_action_meanings()
        if (
            not isinstance(env.action_space, gym.spaces.Discrete)
            or env.action_space.n != 18
            or env.action_space.start != 0
        ):
            raise ValueError("Montezuma requires ALE's eighteen-action set.")
        directions = {
            "UP": "up",
            "DOWN": "down",
            "LEFT": "left",
            "RIGHT": "right",
            "UPRIGHT": "up and right",
            "UPLEFT": "up and left",
            "DOWNRIGHT": "down and right",
            "DOWNLEFT": "down and left",
        }
        descriptions = {
            "NOOP": "Release the controls; ongoing jumps or falls can continue.",
            "FIRE": "Press jump without a direction.",
        }
        for name, direction in directions.items():
            descriptions[name] = (
                f"Hold {direction}; up/down climb when aligned with a ladder or rope."
            )
            descriptions[name + "FIRE"] = f"Hold {direction} and jump."
        if set(meanings) != set(descriptions):
            raise ValueError("Unrecognized Montezuma action meanings.")
        return tuple(Action(name.lower(), i, descriptions[name]) for i, name in enumerate(meanings))

    def encode(self, observation: Any, *, reward: float = 0.0) -> dict:
        ram = np.asarray(observation)
        if ram.shape != (128,) or ram.dtype != np.uint8:
            raise ValueError("Montezuma needs a 128-byte uint8 RAM observation.")
        room, level = int(ram[3]), int(ram[57])
        game_over = int(ram[58]) == 0 and int(ram[126]) == 0x60
        lives = 0 if game_over else (int(ram[58]) & 7) + 1
        inventory = int(ram[65])
        # Only these death-animation states have been calibrated; other animation
        # flags are not presented as proof that the player can accept an action.
        death_animation = int(ram[126]) in (102, 116) or game_over
        epoch = (level, room, lives, death_animation)
        old = self._previous
        previous = old if old and old["epoch"] == epoch and not death_animation else None
        positions = {}

        def actor(identifier, kind, x, y, width, height):
            center = (x + width / 2, y + height / 2)
            positions[identifier] = center
            motion = None
            if previous and identifier in previous["positions"]:
                before = previous["positions"][identifier]
                dx, dy = center[0] - before[0], center[1] - before[1]
                if abs(dx) + abs(dy) <= max(12, 4 * self.frameskip):
                    motion = dict(dx=dx, dy=dy)
            return dict(
                id=identifier,
                kind=kind,
                x=center[0],
                y=center[1],
                width=width,
                height=height,
                motion_per_decision=motion,
            )

        player = actor("player", "player", int(ram[42]) - 1, 308 - int(ram[43]), 8, 20)
        player["facing"] = "left" if int(ram[103]) & 64 else "right"
        player["on_ladder"] = int(ram[94]) == 1
        items, hazards, doors = [], [], []
        lit = room < 16 or bool(inventory & 128)
        kind = int(ram[49])
        object_slot_known = lit or kind == 5 or 8 < kind < 11
        if object_slot_known:
            x = int(ram[44]) - 1
            y = {0: 310, 1: 353, 7: 309}.get(room, 308) - int(ram[45])
            if int(ram[45]) < 216:
                y += 4
            item_types = {
                1: ("ruby", 7, 12),
                2: ("sword", 6, 15),
                3: ("amulet", 6, 15),
                4: ("key", 7, 15),
                6: ("torch", 6, 13),
            }
            if kind in item_types:
                name, w, h = item_types[kind]
                offsets = [0]
                if kind == 1:
                    offsets += [
                        offset for bit, offset in ((1, 16), (2, 32), (4, 64)) if int(ram[84]) & bit
                    ]
                items = [actor(f"{name}_{offset}", name, x + offset, y, w, h) for offset in offsets]
            elif kind in (5, 7, 8, 9, 10):
                name, w, h = (
                    ("skull", 7, 13)
                    if kind == 5
                    else ("snake", 7, 13)
                    if kind in (7, 8)
                    else ("spider", 8, 11)
                )
                if kind in (7, 8):
                    x += 1
                offsets = [0] + [
                    offset for bit, offset in ((1, 16), (2, 32), (4, 64)) if int(ram[84]) & bit
                ]
                hazards = [
                    actor(f"enemy_{offset}", name, x + offset, y, w, h) for offset in offsets
                ]

        # These rooms have an additional skull in a separate RAM slot.
        skull = None
        if room == 1 and int(ram[67]) & 2:
            skull = (int(ram[47]) + 32, 406 - int(ram[46]))
        elif room == 5 and int(ram[69]) & 2:
            skull = (int(ram[47]) + 33, 358 - int(ram[46]))
        elif room == 18 and int(ram[76]) & 32:
            skull = (int(ram[47]) - 1, int(ram[46]) - 147)
        if skull:
            hazards.append(actor("room_skull", "skull", *skull, 7, 13))
        if room in (1, 5, 17):
            for identifier, reg, x, y in (
                ("door_left", 26, 56 if room == 5 else 20, 135 if room == 5 else 54),
                ("door_right", 28, 100 if room == 5 else 136, 135 if room == 5 else 54),
            ):
                doors.append(
                    dict(
                        id=identifier,
                        x=x + 2,
                        y=y + 18.5,
                        width=4,
                        height=37,
                        locked=int(ram[reg]) != 117,
                        requires="key",
                    )
                )
        if room in (0, 7, 12):
            xs = (36, 44, 60, 68, 88, 96, 112, 120) if room == 12 else (16, 36, 44, 112, 120, 140)
            for i, x in enumerate(xs):
                hazards.append(
                    dict(
                        id=f"beam_{i}",
                        kind="beam",
                        x=x + 2,
                        y=73,
                        width=4,
                        height=40,
                        active=int(ram[26]) != 117,
                    )
                )

        geometry = room_geometry(room, ram)
        # Record only crossings actually observed near a matching prior-room exit.
        # Death/respawn or arbitrary teleports do not establish an exit destination.
        if (
            old
            and old["epoch"][:2] != epoch[:2]
            and old["epoch"][2:] == epoch[2:]
            and not death_animation
        ):
            px, py = old["positions"]["player"]
            candidates = [
                ex
                for ex in old["exits"]
                if abs(px - ex["x"]) <= 16 and abs(py + 10 - ex["feet_y"]) <= 24
            ]
            if len(candidates) == 1:
                self._destinations[(*old["epoch"][:2], candidates[0]["id"])] = dict(
                    level=level, room=room
                )
        self._visited.add((level, room))
        if geometry:
            for ex in geometry["exit_candidates"]:
                ex["destination"] = self._destinations.get((level, room, ex["id"]))
        self._previous = dict(
            epoch=epoch,
            positions=positions,
            exits=[] if geometry is None else geometry["exit_candidates"],
        )
        score = sum(
            ((int(ram[i]) >> 4) * 10 + (int(ram[i]) & 15)) * 100 ** (21 - i) for i in (19, 20, 21)
        )
        return {
            "emulator_frames_per_action": self.frameskip,
            "room": {
                "level": level,
                "id": room,
                "lighting": "unknown" if room == 15 else "lit" if lit else "dark",
                "geometry_status": "unknown"
                if geometry is None
                else "unverified_template"
                if room == 15
                else "reference_template",
            },
            "score": score,
            "lives_remaining": lives,
            "game_over": game_over,
            "player": player,
            "inventory": {
                "keys": (inventory & 30).bit_count(),
                "swords": (inventory & 96).bit_count(),
                "torch": bool(inventory & 128),
                "amulet": bool(inventory & 1),
            },
            "items": items if object_slot_known else None,
            "hazards": hazards,
            "doors": doors,
            "geometry": geometry,
            "local_geometry": local_geometry(player, geometry),
            "exploration": {
                "visited_rooms_this_level": sorted(r for lev, r in self._visited if lev == level)
            },
            "description": (
                "Coordinates are pixels within the CURRENT room: x increases right, y down; "
                "actors use center x/y and width/height. Geometry uses left/right and top/bottom "
                "bounds; surface tops are standing foot heights. Motion is measured displacement "
                "per decision, not a forecast. Aligned surfaces are geometric matches, not proof "
                "of groundedness or safety. Up/down climb ladders or ropes when aligned; fire "
                "jumps. Jumps and falls can continue after controls are released. Long falls can "
                "kill even with a platform below. Temporary platforms can vanish; conveyors "
                "move you. Doors require keys; beam active=false means currently off, not safe "
                "for a whole crossing. Exit candidates may be blocked; destinations are recorded "
                "only after observed crossings. Reference geometry can describe dark rooms. "
                "This prototype's objects and geometry may be incomplete or approximate; null "
                "means unknown. Jump reachability, safe drop heights, rope state, conveyor "
                "direction, enemy patrol limits and full animation/control state are not decoded."
            ),
        }
