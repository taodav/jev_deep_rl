"""Static corridor geometry for the four default ALE Ms. Pac-Man mazes.

These tables were measured offline against ALE 0.12.1's standard ROM, including
all four layouts selected by RAM[0]. Runtime observations use RAM only. A cell
is a corridor sample, not an 8x12 solid floor tile. Adjacent non-wall samples
are connected; the ghost house is excluded. Tests compare with rendered walls.
"""

from bisect import bisect_right
from functools import lru_cache
from heapq import heapify, heappop, heappush
from math import hypot

LAYOUTS = (
    (
        "....#........#....",
        ".##.#.######.#.##.",
        "..................",
        "#.#.##.####.##.#.#",
        "..#............#..",
        "#.####.####.####.#",
        "#......####......#",
        "#.####.####.####.#",
        "..#............#..",
        "#.#.#.#.##.#.#.#.#",
        "....#.#....#.#....",
        ".##.#...##...#.##.",
        ".##.###.##.###.##.",
        "..................",
    ),
    (
        "..................",
        ".####.######.####.",
        "..#...#....#...#..",
        "#.#.#.#.##.#.#.#.#",
        "#.#.#........#.#.#",
        "....##.####.##....",
        ".##....####....##.",
        ".##.##.####.##.##.",
        ".##.#........#.##.",
        "......######......",
        "##.##.#....#.##.##",
        "....#...##...#....",
        ".##.#.######.#.##.",
        "....#........#....",
    ),
    (
        "......#....#......",
        ".##.#...##...#.##.",
        "....#.#....#.#....",
        "##.##.######.##.##",
        "..................",
        ".##.##.####.##.##.",
        ".#...#.####.#...#.",
        ".#.#.#.####.#.#.#.",
        "...#..........#...",
        "##.##.#.##.#.##.##",
        "......#....#......",
        ".#.##...##...##.#.",
        ".#.##.#.##.#.##.#.",
        "......#....#......",
    ),
    (
        "..................",
        ".#.##.######.##.#.",
        ".#....#....#....#.",
        ".##.#.#.##.#.#.##.",
        "....#........#....",
        "#.#.##.####.##.#.#",
        "..#....####....#..",
        "###.##.####.##.###",
        "..#.#........#.#..",
        "#.#.#.#.##.#.#.#.#",
        "......#....#......",
        ".#.##.######.##.#.",
        ".#.##.#....#.##.#.",
        "........##........",
    ),
)
TUNNEL_ROWS = ((4, 8), (5, 13), (8,), (6, 8))
X = tuple(9 + 8 * c + (4 if c >= 9 else 0) for c in range(18))
Y = tuple(7.5 + 12 * r for r in range(14))
DIRECTIONS = (("up", -1, 0), ("right", 0, 1), ("down", 1, 0), ("left", 0, -1))
OPPOSITE = {"up": "down", "down": "up", "left": "right", "right": "left"}


def maze_position(x: float, y: float) -> tuple[float, float]:
    """Pixel centers -> fractional row/column, interpolating each actual gap.

    Most horizontal gaps are 8px, the middle gap is 12px, and the tunnel link
    from column 17 to column 0 is 20px. Columns in [17, 18) lie on that link.
    Keep full precision here so motion is computed before presentation rounding.
    """
    row = (y - Y[0]) / 12
    if x < X[0] or x > X[-1]:
        column = (17 + ((x - X[-1]) % 160) / 20) % 18
    else:
        left = min(bisect_right(X, x) - 1, 16)
        column = left + (x - X[left]) / (X[left + 1] - X[left])
    return row, column


def cell_number(value: float) -> int | float:
    """Avoid long repeating decimals and unnecessary .0 in the model input."""
    value = round(float(value), 3)
    return int(value) if value.is_integer() else value


class Maze:
    def __init__(self, layout: int):
        self.rows = LAYOUTS[layout]
        self.tunnel_rows = TUNNEL_ROWS[layout]
        self.graph = {(r, c): {} for r in range(14) for c in range(18) if self.rows[r][c] != "#"}
        for (r, c), neighbors in self.graph.items():
            for direction, dr, dc in DIRECTIONS:
                target = (r + dr, c + dc)
                if r in self.tunnel_rows and dr == 0:
                    target = (r, (c + dc) % 18)
                if target in self.graph:
                    distance = (
                        20
                        if abs(target[1] - c) == 17
                        else abs(X[target[1]] - X[c]) + abs(Y[target[0]] - Y[r])
                    )
                    neighbors[direction] = (target, distance)

    def locate(self, x: float, y: float) -> dict | None:
        """Project to a corridor centerline, keeping both endpoints between cells.

        Two pixels of tolerance allow sprite/centerline rounding. A ghost inside
        the house must not be snapped onto an unrelated player corridor.
        """
        best = None
        for cell, neighbors in self.graph.items():
            r, c = cell
            distance = hypot(x - X[c], y - Y[r])
            if distance <= 0.75:
                return {"cell": cell, "links": dict(neighbors), "offset": round(distance, 1)}
            for direction in ("right", "down"):
                if direction not in neighbors:
                    continue
                target, length = neighbors[direction]
                # The rightward wrap has two equivalent screen projections.
                shifts = (-160, 0) if cell[1] == 17 and target[1] == 0 else (0,)
                for shift in shifts:
                    start_x, start_y = X[c] + shift, Y[r]
                    along = x - start_x if direction == "right" else y - start_y
                    along = max(0, min(length, along))
                    px = start_x + along if direction == "right" else start_x
                    py = start_y + along if direction == "down" else start_y
                    distance = hypot(x - px, y - py)
                    if best is None or distance < best[0]:
                        if along <= 0.75:
                            location = {"cell": cell, "links": dict(self.graph[cell])}
                        elif length - along <= 0.75:
                            location = {"cell": target, "links": dict(self.graph[target])}
                        else:
                            location = {
                                "cell": None,
                                "links": {
                                    OPPOSITE[direction]: (cell, along),
                                    direction: (target, length - along),
                                },
                            }
                        best = (distance, location)
        if best is None or best[0] > 2:
            return None
        return {**best[1], "offset": round(best[0], 1)}

    @staticmethod
    def links_in_cells(location: dict) -> dict:
        """Location links converted from pixel lengths to fractions of an edge."""
        if location["cell"] is not None:
            return {direction: (cell, 1.0) for direction, (cell, _) in location["links"].items()}
        length = sum(distance for _, distance in location["links"].values())
        return {
            direction: (cell, distance / length)
            for direction, (cell, distance) in location["links"].items()
        }

    def distances_in_cells(self, location: dict | None) -> dict:
        """Shortest path by cell connections; one full edge counts as one cell."""
        if location is None:
            return {}
        starts = (
            {location["cell"]: 0}
            if location["cell"] is not None
            else {cell: distance for cell, distance in self.links_in_cells(location).values()}
        )
        result = dict(starts)
        queue = [(distance, cell) for cell, distance in starts.items()]
        # At most two endpoints; heapify is still required for unequal distances.
        heapify(queue)
        while queue:
            distance, cell = heappop(queue)
            if distance != result[cell]:
                continue
            for target, _ in self.graph[cell].values():
                candidate = distance + 1
                if candidate < result.get(target, float("inf")):
                    result[target] = candidate
                    heappush(queue, (candidate, target))
        return result

    def corridors(self, location: dict | None, pellets: set) -> dict | None:
        if location is None:
            return None
        result = {}
        for direction, (target, distance) in self.links_in_cells(location).items():
            count = int(target in pellets)
            seen = set()
            while target not in seen:
                seen.add(target)
                neighbors = self.graph[target]
                if len(neighbors) != 2 or direction not in neighbors:
                    break  # a junction, corner or dead end
                target, _ = neighbors[direction]
                distance += 1
                count += int(target in pellets)
            result[direction] = {
                "next_corner_or_junction": {"row": target[0], "column": target[1]},
                "distance_cells": cell_number(distance),
                "pellets_on_segment": count,
                "exits_there": list(self.graph[target]),
            }
        return result


@lru_cache(maxsize=4)
def maze_for(layout: int) -> Maze:
    if layout not in range(4):
        raise ValueError("Unrecognized Ms. Pac-Man maze layout.")
    return Maze(layout)
