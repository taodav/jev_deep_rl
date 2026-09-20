"""Room-local geometry for a Montezuma descriptor prototype.

Reference: OCAtari's MIT-licensed montezumarevenge RAM decoder. See
THIRD_PARTY_NOTICES.md. Coordinates are screen pixels, right/down positive.
These are geometry templates, NOT certified collision boxes or safe routes.
Dynamic bridges, doors, beams and actors are decoded separately.
"""


def layout(platforms, ladders=(), walls=(), ropes=(), conveyors=()):
    return dict(platforms=platforms, ladders=ladders, walls=walls, ropes=ropes, conveyors=conveyors)


# Rectangles are (left, top, width, height); their right/bottom bounds are exclusive.
ROOMS = {
    0: layout([(4, 94, 154, 1)], [(72, 94, 16, 102)], [(0, 53, 4, 42)]),
    1: layout(
        [
            (0, 93, 56, 2),
            (104, 93, 56, 2),
            (68, 93, 23, 2),
            (76, 136, 8, 5),
            (8, 136, 28, 2),
            (124, 136, 28, 2),
            (16, 180, 128, 1),
        ],
        [(72, 93, 16, 42), (16, 136, 16, 45), (128, 136, 16, 45)],
        [(0, 96, 8, 40), (0, 136, 16, 45), (152, 96, 8, 40), (144, 136, 16, 45)],
        [(112, 96, 1, 39)],
        [(60, 136, 15, 5), (85, 136, 15, 5)],
    ),
    2: layout([(4, 93, 154, 1)], [(72, 93, 16, 102)], [(156, 52, 4, 42)]),
    3: layout([(4, 93, 154, 1)], [(72, 93, 16, 102)], [(0, 53, 4, 41)]),
    4: layout([(4, 93, 154, 1)], [(72, 52, 16, 39), (72, 93, 16, 102)]),
    5: layout(
        [(0, 93, 48, 2), (112, 93, 48, 2), (48, 130, 64, 3), (4, 171, 154, 1), (76, 93, 8, 5)],
        [(72, 171, 16, 26)],
        [(32, 53, 4, 40), (124, 53, 4, 40), (0, 96, 4, 77), (156, 96, 4, 77)],
        [(41, 97, 1, 24), (125, 95, 2, 52)],
        [(60, 93, 15, 5), (85, 93, 15, 5)],
    ),
    6: layout([(0, 93, 160, 1)], [(72, 53, 16, 38)]),
    7: layout([(4, 94, 154, 1)], [(72, 94, 16, 102)], [(156, 53, 4, 42)]),
    8: layout(
        [
            (4, 143, 12, 4),
            (144, 143, 12, 4),
            (4, 93, 43, 3),
            (76, 93, 8, 3),
            (112, 93, 48, 3),
            (4, 173, 152, 1),
        ],
        walls=[(0, 53, 4, 121), (156, 97, 4, 77)],
        ropes=[(80, 96, 1, 51)],
    ),
    9: layout([(0, 93, 158, 1)], [(72, 53, 16, 38)], [(156, 52, 3, 41)]),
    10: layout([(0, 93, 36, 1), (124, 93, 36, 1)], [(72, 53, 16, 38)], [(0, 53, 3, 40)]),
    11: layout([(0, 93, 160, 1)], [(72, 52, 16, 39), (72, 93, 16, 102)]),
    12: layout([(0, 94, 160, 1)]),
    13: layout([(0, 93, 160, 1)], [(72, 52, 16, 39), (72, 93, 16, 102)]),
    14: layout(
        [(0, 93, 40, 3), (68, 93, 24, 3), (120, 93, 40, 3), (16, 168, 128, 1)],
        [(72, 169, 16, 25)],
        [(0, 97, 16, 73), (144, 97, 16, 73)],
        [(71, 96, 1, 48), (87, 97, 2, 33)],
    ),
    15: layout([(0, 94, 160, 1)]),
    16: layout([(0, 93, 160, 1)]),
    17: layout([(0, 94, 160, 1)]),
    18: layout([(0, 94, 36, 1), (124, 94, 36, 1)]),
    19: layout([(0, 93, 160, 1)], [(72, 53, 16, 38)]),
    20: layout([(0, 94, 36, 1), (124, 94, 32, 1)], walls=[(156, 52, 3, 42)]),
    21: layout([(0, 93, 158, 1)], [(72, 53, 16, 38)], [(0, 53, 3, 40)]),
    22: layout([(0, 93, 36, 1), (124, 93, 36, 1)], [(72, 53, 16, 38)]),
    23: layout([(0, 93, 158, 1)], walls=[(156, 53, 3, 40)]),
}


def rectangle(identifier, bounds):
    x, y, width, height = bounds
    return {"id": identifier, "x_min": x, "x_max": x + width, "y_top": y, "y_bottom": y + height}


def room_geometry(room: int, ram) -> dict | None:
    """Return fresh geometry; unknown rooms stay unknown, never reuse old geometry."""
    if room not in ROOMS:
        return None
    result = {
        name: [rectangle(f"{name.rstrip('s')}_{i}", box) for i, box in enumerate(boxes)]
        for name, boxes in ROOMS[room].items()
    }
    temporary = []
    if room in (10, 18, 20, 22):
        y = 93 if room in (10, 22) else 94
        hidden_value = 214 if room >= 16 and not int(ram[65]) & 128 else 232
        temporary = [(36, y, 88, 7, int(ram[34]) != hidden_value)]
    elif room == 8:
        temporary = [
            (x, y, 12, 4, int(ram[34]) != 144)
            for x in (4, 144)
            for y in (103, 113, 123, 133, 153, 163)
        ]
    result["temporary_platforms"] = [
        {**rectangle(f"temporary_{i}", b[:4]), "present": b[4]} for i, b in enumerate(temporary)
    ]
    # Surface continuations at the room boundary are exit candidates. Destination
    # and actual traversability are unknown until observed; doors may block them.
    exits = {}
    for p in result["platforms"]:
        for side, x in (("left", 0), ("right", 160)):
            reaches_edge = p["x_min"] <= 4 if side == "left" else p["x_max"] >= 156
            blocked = any(
                w["x_min"] <= x + 4
                and w["x_max"] >= x - 4
                and w["y_top"] < p["y_top"]
                and w["y_bottom"] >= p["y_top"] - 20
                for w in result["walls"]
            )
            if reaches_edge and not blocked:
                identifier = f"{side}_{p['y_top']}"
                exits[identifier] = dict(
                    id=identifier, direction=side, x=x, feet_y=p["y_top"], destination=None
                )
    for ladder in result["ladders"]:
        for side, boundary, near in (
            ("up", 53, ladder["y_top"] <= 54),
            ("down", 195, ladder["y_bottom"] >= 194),
        ):
            if near:
                identifier = f"{side}_{ladder['id']}"
                exits[identifier] = dict(
                    id=identifier,
                    direction=side,
                    x=(ladder["x_min"] + ladder["x_max"]) / 2,
                    feet_y=boundary,
                    destination=None,
                )
    result["exit_candidates"] = list(exits.values())
    return result


def local_geometry(player, geometry):
    """Useful geometric relations, without asserting jump reachability/safety."""
    if geometry is None:
        return None
    x, feet = player["x"], player["y"] + player["height"] / 2
    surfaces = (
        geometry["platforms"]
        + geometry["conveyors"]
        + [p for p in geometry["temporary_platforms"] if p["present"]]
    )
    support = [p for p in surfaces if p["x_min"] <= x <= p["x_max"] and abs(feet - p["y_top"]) <= 2]
    ladders = sorted(geometry["ladders"], key=lambda p: abs(x - (p["x_min"] + p["x_max"]) / 2))
    return {
        "feet_y": feet,
        "aligned_surface_ids": [p["id"] for p in support],
        "surface_edges_dx": [
            {"id": p["id"], "left": p["x_min"] - x, "right": p["x_max"] - x} for p in support
        ],
        "ladders": [
            {
                "id": p["id"],
                "center_dx": (p["x_min"] + p["x_max"]) / 2 - x,
                "top_dy": p["y_top"] - feet,
                "bottom_dy": p["y_bottom"] - feet,
            }
            for p in ladders
        ],
    }
