"""Pure unit tests for the grid planner. No ROS, no sim -- runs anywhere:

    python3 bridge/tests/test_nav.py
"""

import json
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import nav  # noqa: E402


def box(name, x, y, w, d, h=0.8, yaw=0.0):
    return {"id": name, "name": name, "kind": "box",
            "x": x, "y": y, "z": 0, "w": w, "d": d, "h": h, "yaw": yaw}


def s_maze():
    return [box("top", 4, 2.1, 6.4, 0.2), box("bot", 4, -2.1, 6.4, 0.2),
            box("right", 7.1, 0, 0.2, 4.4), box("ba", 3.0, -0.55, 0.2, 3.3),
            box("bb", 5.0, 0.55, 0.2, 3.3)]


def run():
    # footprint distance: surface, not center
    wall = box("w", 3.0, 0.0, 0.2, 6.0)
    assert abs(nav.footprint_distance(0.0, 0.0, wall) - 2.9) < 1e-6
    assert nav.footprint_distance(3.0, 0.0, wall) == 0.0
    ball = {"id": "b", "kind": "sphere", "x": 2, "y": 0, "z": 0, "w": 1, "d": 1, "h": 1}
    assert abs(nav.footprint_distance(0.0, 0.0, ball) - 1.5) < 1e-6
    rot = box("r", 2.0, 0.0, 4.0, 0.2, yaw=90.0)  # long axis now along Y
    assert abs(nav.footprint_distance(0.0, 0.0, rot) - 1.9) < 1e-2

    # rotated footprints must block correctly: a 45-degree wall between start and goal
    diag = [box("diag", 2.0, 0.0, 6.0, 0.2, yaw=45.0)]
    route = nav.plan_path(diag, (0, 0), (4.0, 0.0), inflate=0.3)
    assert route is not None and len(route) >= 2  # must route around an end

    # the S maze solves; a fully sealed box does not
    route = nav.plan_path(s_maze(), (0, 0), (6.3, 0.0), inflate=0.28, robot_height=0.45)
    assert route is not None
    sealed = s_maze() + [box("left", 0.9, 0, 0.2, 4.4)]
    assert nav.plan_path(sealed, (-0.5, 0), (6.3, 0.0), inflate=0.28,
                         robot_height=0.45) is None

    # objects the robot fits under do not block
    bridge_deck = [dict(box("deck", 2, 0, 2, 2), z=0.8, h=0.2)]
    assert nav.plan_path(bridge_deck, (0, 0), (4, 0), inflate=0.28,
                         robot_height=0.45) == [(4, 0)]

    # target-object mode ends near the footprint, not inside it
    approach = nav.plan_path([wall], (0, 0), (3.0, 0.0), inflate=0.28,
                             target_obj=wall, stop_distance=0.5)
    assert approach is not None
    gap = nav.footprint_distance(approach[-1][0], approach[-1][1], wall)
    assert 0.2 <= gap <= 0.95, gap

    # the captured user maze is reachable only via the outside route; verify both facts
    fixture = os.path.join(os.path.dirname(__file__), "fixtures", "user_maze.json")
    world = json.load(open(fixture))
    route = nav.plan_path(world, (0, 0), (7.0, 0.0), inflate=0.28, robot_height=0.45)
    assert route is not None
    length = sum(math.dist(a, b) for a, b in zip([(0, 0)] + route, route))
    assert length > 12.0, length  # must be the long way around, not through a wall

    print("test_nav: all assertions passed")


if __name__ == "__main__":
    run()
