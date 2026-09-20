"""Deterministic grid path planning over the authored world list.

The world is fully known (same source of truth the synthesized scan raycasts), so
navigation is a solved geometry problem: rasterize footprints inflated by the robot's
radius, A* to the goal, smooth to a few waypoints. No ROS imports -- testable anywhere.

Also doubles as a world validator: `plan_path` returning None is exactly "the robot
cannot physically get there", which catches LLM-built mazes with no entrance.
"""

import heapq
import math
from typing import Dict, List, Optional, Sequence, Tuple

CELL = 0.1          # m per grid cell
MARGIN = 1.0        # free border around everything, m
MAX_CELLS = 700     # per axis; cell size grows for gigantic worlds instead of failing
SQRT2 = math.sqrt(2.0)

Point = Tuple[float, float]


def footprint_distance(px: float, py: float, obj: Dict) -> float:
    """Distance from a point to the object's 2D footprint edge (0 inside).

    This is what "how far am I from the wall" must mean: measured to the surface,
    never to the center -- a 6 m wall's center is 3 m from its own edge.
    """
    if obj.get("kind") in ("cylinder", "cone", "sphere"):
        radius = max(float(obj["w"]), float(obj["d"])) / 2.0
        return max(0.0, math.hypot(px - float(obj["x"]), py - float(obj["y"])) - radius)
    yaw = math.radians(float(obj.get("yaw", 0.0)))
    dx, dy = px - float(obj["x"]), py - float(obj["y"])
    lx = dx * math.cos(-yaw) - dy * math.sin(-yaw)
    ly = dx * math.sin(-yaw) + dy * math.cos(-yaw)
    qx = max(abs(lx) - float(obj["w"]) / 2.0, 0.0)
    qy = max(abs(ly) - float(obj["d"]) / 2.0, 0.0)
    return math.hypot(qx, qy)


def _blocks_robot(obj: Dict, robot_height: float) -> bool:
    """An object only blocks driving if its height range overlaps the robot's body."""
    bottom = float(obj.get("z", 0.0))
    top = bottom + float(obj.get("h", 0.0))
    return bottom <= robot_height and top >= 0.03


class _Grid:
    def __init__(self, world: Sequence[Dict], extra: Sequence[Point],
                 inflate: float, robot_height: float) -> None:
        xs = [p[0] for p in extra]
        ys = [p[1] for p in extra]
        for o in world:
            reach = max(float(o["w"]), float(o["d"])) / 2.0 + inflate
            xs += [float(o["x"]) - reach, float(o["x"]) + reach]
            ys += [float(o["y"]) - reach, float(o["y"]) + reach]
        self.min_x = min(xs) - MARGIN
        self.min_y = min(ys) - MARGIN
        span = max(max(xs) - self.min_x, max(ys) - self.min_y) + MARGIN
        self.cell = max(CELL, span / MAX_CELLS)
        self.nx = int((max(xs) + MARGIN - self.min_x) / self.cell) + 1
        self.ny = int((max(ys) + MARGIN - self.min_y) / self.cell) + 1
        self.blocked = bytearray(self.nx * self.ny)
        for o in world:
            if not _blocks_robot(o, robot_height):
                continue
            self._rasterize(o, inflate)

    def _rasterize(self, obj: Dict, inflate: float) -> None:
        reach = max(float(obj["w"]), float(obj["d"])) / 2.0 + inflate + self.cell
        ci0, cj0 = self.to_cell(float(obj["x"]) - reach, float(obj["y"]) - reach)
        ci1, cj1 = self.to_cell(float(obj["x"]) + reach, float(obj["y"]) + reach)
        for j in range(max(0, cj0), min(self.ny, cj1 + 1)):
            for i in range(max(0, ci0), min(self.nx, ci1 + 1)):
                x, y = self.to_world(i, j)
                if footprint_distance(x, y, obj) <= inflate:
                    self.blocked[j * self.nx + i] = 1

    def to_cell(self, x: float, y: float) -> Tuple[int, int]:
        return int((x - self.min_x) / self.cell), int((y - self.min_y) / self.cell)

    def to_world(self, i: int, j: int) -> Point:
        return (self.min_x + (i + 0.5) * self.cell, self.min_y + (j + 0.5) * self.cell)

    def free(self, i: int, j: int) -> bool:
        return 0 <= i < self.nx and 0 <= j < self.ny and not self.blocked[j * self.nx + i]

    def nearest_free(self, i: int, j: int, max_radius_m: float) -> Optional[Tuple[int, int]]:
        """Closest free cell by expanding rings -- rescues a start pose that sits
        inside the inflation band (robot parked close to a wall)."""
        if self.free(i, j):
            return (i, j)
        max_r = int(max_radius_m / self.cell) + 1
        for r in range(1, max_r + 1):
            for di in range(-r, r + 1):
                for dj in (-r, r):
                    if self.free(i + di, j + dj):
                        return (i + di, j + dj)
            for dj in range(-r + 1, r):
                for di in (-r, r):
                    if self.free(i + di, j + dj):
                        return (i + di, j + dj)
        return None

    def line_free(self, a: Tuple[int, int], b: Tuple[int, int]) -> bool:
        """Supercover walk from a to b: every touched cell must be free."""
        (i0, j0), (i1, j1) = a, b
        di, dj = abs(i1 - i0), abs(j1 - j0)
        si = 1 if i1 > i0 else -1
        sj = 1 if j1 > j0 else -1
        err = di - dj
        i, j = i0, j0
        while True:
            if not self.free(i, j):
                return False
            if (i, j) == (i1, j1):
                return True
            e2 = 2 * err
            if e2 > -dj:
                err -= dj
                i += si
            if e2 < di:
                err += di
                j += sj


def _astar(grid: _Grid, start: Tuple[int, int], goals: set) -> Optional[List[Tuple[int, int]]]:
    goal_list = list(goals)
    gx = sum(g[0] for g in goal_list) / len(goal_list)
    gy = sum(g[1] for g in goal_list) / len(goal_list)

    def h(cell):
        return math.hypot(cell[0] - gx, cell[1] - gy)

    open_heap = [(h(start), 0.0, start)]
    came: Dict[Tuple[int, int], Tuple[int, int]] = {}
    best_g = {start: 0.0}
    while open_heap:
        _f, g, cur = heapq.heappop(open_heap)
        if cur in goals:
            path = [cur]
            while cur in came:
                cur = came[cur]
                path.append(cur)
            return path[::-1]
        if g > best_g.get(cur, math.inf):
            continue
        ci, cj = cur
        for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)):
            ni, nj = ci + di, cj + dj
            if not grid.free(ni, nj):
                continue
            if di and dj and not (grid.free(ci + di, cj) and grid.free(ci, cj + dj)):
                continue  # no cutting corners through a blocked diagonal
            ng = g + (SQRT2 if di and dj else 1.0)
            if ng < best_g.get((ni, nj), math.inf):
                best_g[(ni, nj)] = ng
                came[(ni, nj)] = cur
                heapq.heappush(open_heap, (ng + h((ni, nj)), ng, (ni, nj)))
    return None


def _smooth(grid: _Grid, cells: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    """Greedy line-of-sight shortcutting: keep only the corners that matter."""
    if len(cells) <= 2:
        return cells
    out = [cells[0]]
    idx = 0
    while idx < len(cells) - 1:
        nxt = idx + 1
        for probe in range(len(cells) - 1, idx, -1):
            if grid.line_free(cells[idx], cells[probe]):
                nxt = probe
                break
        out.append(cells[nxt])
        idx = nxt
    return out


def plan_path(world: Sequence[Dict], start: Point, goal: Point, inflate: float,
              robot_height: float = 0.5, target_obj: Optional[Dict] = None,
              stop_distance: float = 0.0, ignore_ids: Sequence[str] = ()) -> Optional[List[Point]]:
    """Waypoints from start toward goal, or None when no traversable route exists.

    With `target_obj` the route may end anywhere the arrival condition holds --
    within `stop_distance` (plus a small approach band) of the object's FOOTPRINT --
    so "go to the wall" routes to whichever face of the wall is reachable.
    """
    obstacles = [o for o in world if o.get("id") not in set(ignore_ids)]
    if not obstacles:
        return [goal]
    grid = _Grid(obstacles, [start, goal], inflate, robot_height)

    start_cell = grid.nearest_free(*grid.to_cell(*start), max_radius_m=inflate + 0.6)
    if start_cell is None:
        return None

    goals: set = set()
    if target_obj is not None:
        band = max(stop_distance, 0.05) + inflate + grid.cell
        reach = max(float(target_obj["w"]), float(target_obj["d"])) / 2.0 + band + grid.cell
        ci0, cj0 = grid.to_cell(float(target_obj["x"]) - reach, float(target_obj["y"]) - reach)
        ci1, cj1 = grid.to_cell(float(target_obj["x"]) + reach, float(target_obj["y"]) + reach)
        for j in range(max(0, cj0), min(grid.ny, cj1 + 1)):
            for i in range(max(0, ci0), min(grid.nx, ci1 + 1)):
                if not grid.free(i, j):
                    continue
                x, y = grid.to_world(i, j)
                if footprint_distance(x, y, target_obj) <= band:
                    goals.add((i, j))
    if not goals:
        goal_cell = grid.nearest_free(*grid.to_cell(*goal), max_radius_m=inflate + 0.6)
        if goal_cell is None:
            return None
        goals = {goal_cell}

    cells = _astar(grid, start_cell, goals)
    if cells is None:
        return None
    waypoints = [grid.to_world(i, j) for (i, j) in _smooth(grid, cells)]
    if target_obj is None:
        # The exact requested point is the true final target when it is free space.
        end = waypoints[-1]
        if math.hypot(end[0] - goal[0], end[1] - goal[1]) <= 2 * grid.cell:
            waypoints[-1] = goal
    if len(waypoints) > 1:
        waypoints = waypoints[1:]  # drop the start cell; the controller is already there
    return waypoints


def reachable(world: Sequence[Dict], start: Point, goal: Point, inflate: float,
              robot_height: float = 0.5) -> bool:
    return plan_path(world, start, goal, inflate, robot_height) is not None
