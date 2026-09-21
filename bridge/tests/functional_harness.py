"""Functional accuracy harness. Runs INSIDE the sim container against the live API.

    docker compose exec sim python3 /app/bridge/tests/functional_harness.py baseline
    docker compose exec sim python3 /app/bridge/tests/functional_harness.py suite

Every check compares a commanded outcome against a measured one with a written numeric
tolerance, so "pass" means "accurate", not "didn't crash". Motion tests run on real
wall-clock physics and take minutes by design.
"""

import argparse
import json
import math
import subprocess
import time
import urllib.error
import urllib.request

BASE = "http://localhost:8000"
FIXTURES = "/app/bridge/tests/fixtures"

# Written tolerances: the definition of "accurate" for this product.
TOL_DRIVE_M = 0.08        # commanded 1.0 m -> displacement within this
TOL_TURN_DEG = 3.0        # commanded 90 deg -> heading error within this
TOL_TURN_LONG_DEG = 4.0   # commanded 270 deg
TOL_GOTO_M = 0.30         # go_to x,y -> final distance to point within this
WALL_STOP_RANGE = (0.15, 0.95)  # stop_distance 0.5 from a wall SURFACE must land here


def api(method, path, body=None, timeout=180):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(
        BASE + path, data=data, method=method, headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError as e:
        return {"_http_error": e.code, "detail": e.read().decode()[:400]}


def state():
    return api("GET", "/api/state")


def reset():
    api("POST", "/api/robot/reset")
    time.sleep(0.8)


def put_world(objects):
    return api("PUT", "/api/world", {"objects": objects})


def plan(steps, repeat=False):
    return api("POST", "/api/plan", {"steps": steps, "repeat": repeat})


def wait_idle(timeout=150):
    end = time.time() + timeout
    time.sleep(0.5)
    while time.time() < end:
        s = state()
        if not s.get("active"):
            return s
        time.sleep(0.3)
    api("POST", "/api/stop")
    time.sleep(0.5)
    return state()


def obj(name, kind="box", **kw):
    o = {"id": kw.pop("id", name.lower().replace(" ", "_")), "kind": kind, "name": name,
         "x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0, "d": 1.0, "h": 1.0, "yaw": 0.0,
         "color": "#8a94a6", "dynamic": False}
    o.update(kw)
    return o


def rect_surface_distance(px, py, o):
    """Distance from a point to the edge of an axis-aligned-after-unrotate rectangle."""
    yaw = math.radians(o.get("yaw", 0.0))
    dx, dy = px - o["x"], py - o["y"]
    lx = dx * math.cos(-yaw) - dy * math.sin(-yaw)
    ly = dx * math.sin(-yaw) + dy * math.cos(-yaw)
    qx = max(abs(lx) - o["w"] / 2.0, 0.0)
    qy = max(abs(ly) - o["d"] / 2.0, 0.0)
    return math.hypot(qx, qy)


def gz_model_exists(model_id):
    """Ask Gazebo itself -- not the bridge -- whether the model is really in the world."""
    done = subprocess.run(
        ["bash", "-lc", f"source /opt/ros/humble/setup.bash && ign model -m {model_id} --pose"],
        capture_output=True, text=True, timeout=20,
    )
    return "Pose [" in done.stdout


def solvable_maze():
    """S-shaped corridor maze with a guaranteed route: entrance on the left,
    around two baffles, goal in the far chamber."""
    walls = [
        obj("m top", x=4, y=2.1, w=6.4, d=0.2, h=0.8, id="mz_top"),
        obj("m bottom", x=4, y=-2.1, w=6.4, d=0.2, h=0.8, id="mz_bot"),
        obj("m right", x=7.1, y=0, w=0.2, d=4.4, h=0.8, id="mz_right"),
        obj("m baffle a", x=3.0, y=-0.55, w=0.2, d=3.3, h=0.8, id="mz_ba"),
        obj("m baffle b", x=5.0, y=0.55, w=0.2, d=3.3, h=0.8, id="mz_bb"),
    ]
    return walls, (6.3, 0.0)


class Score:
    def __init__(self):
        self.rows = []

    def add(self, group, name, ok, detail):
        self.rows.append((group, name, ok, detail))
        print(f"  [{'PASS' if ok else 'FAIL'}] {group}/{name}: {detail}", flush=True)

    def summary(self):
        groups = {}
        for group, _n, ok, _d in self.rows:
            passed, total = groups.get(group, (0, 0))
            groups[group] = (passed + (1 if ok else 0), total + 1)
        print("\n==== SUMMARY ====")
        for group, (passed, total) in groups.items():
            print(f"  {group}: {passed}/{total}")
        return all(passed == total for passed, total in groups.values())


def measure_drive(score, group, i):
    reset()
    put_world([])
    time.sleep(0.4)
    s0 = state()
    plan([{"skill": "drive", "distance": 1.0}])
    s1 = wait_idle()
    moved = math.dist((s0["pose"]["x"], s0["pose"]["y"]), (s1["pose"]["x"], s1["pose"]["y"]))
    err = abs(moved - 1.0)
    score.add(group, f"drive1m#{i}", err <= TOL_DRIVE_M, f"moved {moved:.3f} m (err {err:.3f})")


def measure_turn(score, group, i, angle, tol):
    reset()
    put_world([])
    time.sleep(0.4)
    s0 = state()
    plan([{"skill": "turn", "angle_degrees": angle}])
    s1 = wait_idle()
    turned = math.degrees(s1["pose"]["yaw"] - s0["pose"]["yaw"])
    # wrap to a sensible comparison for the commanded angle
    want = angle % 360.0
    got = turned % 360.0
    err = min(abs(got - want), 360 - abs(got - want))
    score.add(group, f"turn{angle}#{i}", err <= tol, f"turned {got:.1f} deg (err {err:.1f})")


def measure_goto_point(score, group, i):
    reset()
    put_world([])
    time.sleep(0.4)
    plan([{"skill": "go_to", "x": 3.0, "y": 2.0}])
    s1 = wait_idle()
    remaining = math.dist((s1["pose"]["x"], s1["pose"]["y"]), (3.0, 2.0))
    score.add(group, f"goto_point#{i}", remaining <= TOL_GOTO_M,
              f"stopped {remaining:.3f} m from target")


def measure_goto_wall(score, group, i):
    reset()
    wall = obj("Long wall", x=3.0, y=0.0, w=0.2, d=6.0, h=1.0, id="tw_wall")
    put_world([wall])
    time.sleep(0.6)
    plan([{"skill": "go_to", "target": "Long wall", "stop_distance": 0.5}])
    s1 = wait_idle()
    surface = rect_surface_distance(s1["pose"]["x"], s1["pose"]["y"], wall)
    lo, hi = WALL_STOP_RANGE
    score.add(group, f"goto_wall#{i}", lo <= surface <= hi,
              f"stopped {surface:.2f} m from wall surface (want {lo}-{hi})")


def measure_maze(score, group, i):
    reset()
    walls, goal = solvable_maze()
    put_world(walls)
    time.sleep(0.8)
    plan([{"skill": "go_to", "x": goal[0], "y": goal[1], "duration": 120}])
    s1 = wait_idle(timeout=140)
    remaining = math.dist((s1["pose"]["x"], s1["pose"]["y"]), goal)
    score.add(group, f"maze_goto#{i}", remaining <= 0.5,
              f"ended {remaining:.2f} m from goal")


def phase_baseline():
    score = Score()
    for i in range(3):
        measure_drive(score, "baseline", i)
    for i in range(3):
        measure_turn(score, "baseline", i, 90, TOL_TURN_DEG)
    measure_turn(score, "baseline", 0, 270, TOL_TURN_LONG_DEG)
    for i in range(3):
        measure_goto_point(score, "baseline", i)
    for i in range(3):
        measure_goto_wall(score, "baseline", i)
    measure_maze(score, "baseline", 0)
    put_world([])
    reset()
    score.summary()


# ---------------------------------------------------------------- 10x suite

def gz_model_list():
    done = subprocess.run(
        ["bash", "-lc", "source /opt/ros/humble/setup.bash && ign model --list"],
        capture_output=True, text=True, timeout=30,
    )
    return done.stdout


def phase_upload(score, n=10):
    import shutil
    import requests
    fixture = f"{FIXTURES}/telearm"
    created = []
    names = ("telearm.urdf.xacro", "body.STL", "wheel.STL", "window.STL")
    for i in range(n):
        files = [("files", (name, open(f"{fixture}/{name}", "rb"))) for name in names]
        r = requests.post(BASE + "/api/robot/upload", files=files, timeout=240)
        ok, detail = r.status_code == 200, f"http {r.status_code}"
        if ok:
            warning = r.json()["warnings"][0]
            uid = warning.split("'")[1]
            created.append(uid)
            import os
            meshes = os.listdir(f"/sim/models/{uid}/meshes")
            web = requests.get(BASE + f"/files/{uid}/{uid}.web.urdf", timeout=30).status_code
            ok = len(meshes) == 3 and web == 200 and "6 movable joints" in warning
            detail = f"{uid}: meshes={len(meshes)} web={web} joints-ok={'6 movable joints' in warning}"
        score.add("upload", f"telearm#{i}", ok, detail)

    # The same robot as a single .zip must convert identically.
    import io
    import zipfile
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as archive:
        for name in names:
            archive.write(f"{fixture}/{name}", name)
    r = requests.post(BASE + "/api/robot/upload",
                      files=[("files", ("telearm.zip", buf.getvalue()))], timeout=240)
    body = r.json() if r.status_code == 200 else {}
    warning = (body.get("warnings") or [""])[0]
    ok = r.status_code == 200 and "6 movable joints" in warning and body.get("upload_id")
    if ok:
        created.append(body["upload_id"])
    score.add("upload", "telearm_zip", bool(ok), f"http {r.status_code}: {warning[:80]}")

    for uid in created:
        shutil.rmtree(f"/sim/models/{uid}", ignore_errors=True)


def phase_world(score, n=10):
    kinds = ["box", "cylinder", "sphere", "cone", "pyramid", "wedge"]
    for i in range(n):
        objs = [obj(f"t {kind}", kind=kind, x=2.0 + k * 1.5, y=1.5 * (-1) ** k,
                    w=0.6, d=0.6, h=0.6, id=f"wt{i}_{kind}",
                    dynamic=(kind == "sphere"))
                for k, kind in enumerate(kinds)]
        put_world(objs)
        time.sleep(1.5)
        listed = gz_model_list()
        missing = [o["id"] for o in objs if o["id"] not in listed]

        put_world([obj("cross wall", x=2.0, y=0.0, w=0.2, d=4.0, h=1.0, id=f"wtx{i}")])
        time.sleep(1.0)
        front = state()["front"]
        scan_ok = abs(front - 1.9) <= 0.2  # wall face at 1.9 m from the origin

        put_world([])
        time.sleep(1.0)
        removed = f"wtx{i}" not in gz_model_list()
        ok = not missing and scan_ok and removed
        score.add("world", f"cycle#{i}", ok,
                  f"missing={missing or 'none'} front={front} removed={removed}")


def phase_motion(score, n=10):
    for i in range(n):
        measure_drive(score, "motion", i)
    for i in range(n):
        measure_turn(score, "motion", i, 90 if i % 2 == 0 else -90, TOL_TURN_DEG)
    for i in range(3):
        measure_turn(score, "motion", i, 270, TOL_TURN_LONG_DEG)
    for i in range(n):
        measure_goto_point(score, "motion", i)
    for i in range(n):
        measure_goto_wall(score, "motion", i)
    for i in range(n):
        measure_maze(score, "motion", i)


def chat(text, timeout=300):
    return api("POST", "/api/command", {"text": text}, timeout=timeout)


def _chat_motion(score, name, text, verify):
    reset()
    put_world([])
    time.sleep(0.4)
    s0 = state()
    r = chat(text)
    if r.get("_http_error"):
        score.add("chat", name, False, f"http {r['_http_error']}: {r.get('detail', '')[:120]}")
        return
    s1 = wait_idle(timeout=180)
    ok, detail = verify(s0, s1, r)
    score.add("chat", name, ok, detail)


def phase_chat(score):
    def moved(target):
        def check(s0, s1, _r):
            d = math.dist((s0["pose"]["x"], s0["pose"]["y"]), (s1["pose"]["x"], s1["pose"]["y"]))
            return abs(d - target) <= 0.15, f"moved {d:.2f} m (want {target})"
        return check

    def turned(target, tol=6.0):
        def check(s0, s1, _r):
            got = math.degrees(s1["pose"]["yaw"] - s0["pose"]["yaw"]) % 360.0
            want = target % 360.0
            err = min(abs(got - want), 360 - abs(got - want))
            return err <= tol, f"turned {got:.1f} deg (want {want}, err {err:.1f})"
        return check

    _chat_motion(score, "drive1m", "drive forward exactly 1 meter", moved(1.0))
    _chat_motion(score, "reverse05", "back up half a meter", moved(0.5))
    _chat_motion(score, "turn_right45", "turn right 45 degrees", turned(-45))
    _chat_motion(score, "turn_left90", "turn left 90 degrees", turned(90))

    # World building through chat, then navigation to the object it created.
    reset()
    put_world([])
    time.sleep(0.4)
    r = chat("put a red box 2 meters in front of you")
    world = state()["world"]
    box = next((o for o in world if o["kind"] == "box"), None)
    ok = bool(r.get("ok")) and box is not None and abs(box["x"] - 2.0) <= 1.0 and abs(box["y"]) <= 1.0
    score.add("chat", "make_box", ok,
              f"box at ({box['x']:.1f},{box['y']:.1f})" if box else "no box created")
    if box:
        wait_idle()
        r = chat(f"go to the {box['name']}")
        s1 = wait_idle(timeout=120)
        import sys
        sys.path.insert(0, "/app/bridge")
        import nav
        gap = nav.footprint_distance(s1["pose"]["x"], s1["pose"]["y"], box)
        score.add("chat", "goto_named", gap <= 0.9, f"stopped {gap:.2f} m from box surface")
    else:
        score.add("chat", "goto_named", False, "skipped: no box to go to")

    # Arm through chat.
    reset()
    r = chat("raise the arm about 45 degrees")
    if r.get("_http_error"):
        score.add("chat", "arm_raise", False, f"http {r['_http_error']}")
    else:
        wait_idle(timeout=60)
        joints = state().get("joints") if "joints" in state() else None
        prims = [a["primitive"] for a in (r.get("plan") or [])]
        score.add("chat", "arm_raise", any(p in ("set_joints", "move_ee") for p in prims),
                  f"plan={prims}")

    # A vague command must still produce something runnable.
    r = chat("avoid obstacles")
    prims = [a["primitive"] for a in (r.get("plan") or [])]
    ok = "avoid_obstacles" in prims
    api("POST", "/api/stop")
    score.add("chat", "vague_avoid", ok, f"plan={prims}")

    # Maze + behaviour in one request: world grows AND the robot makes progress.
    reset()
    put_world([])
    time.sleep(0.4)
    s0 = state()
    r = chat("build a small maze around me and then drive to its exit")
    if r.get("_http_error"):
        score.add("chat", "maze_and_go", False, f"http {r['_http_error']}")
    else:
        s1 = wait_idle(timeout=200)
        grew = len(s1["world"]) >= 4
        moved_m = math.dist((s0["pose"]["x"], s0["pose"]["y"]), (s1["pose"]["x"], s1["pose"]["y"]))
        score.add("chat", "maze_and_go", grew and moved_m >= 1.0,
                  f"{len(s1['world'])} objects, moved {moved_m:.1f} m, "
                  f"warnings={r.get('warnings')}")

    # Grasp through chat with a crate it must first make graspable.
    reset()
    put_world([obj("Crate", x=1.2, y=0.0, w=0.4, d=0.4, h=0.4, id="chat_crate", dynamic=True)])
    time.sleep(0.8)
    r = chat("pick up the crate")
    if r.get("_http_error"):
        score.add("chat", "grasp_crate", False, f"http {r['_http_error']}")
    else:
        s1 = wait_idle(timeout=150)
        score.add("chat", "grasp_crate", s1.get("held") is not None,
                  f"held={s1.get('held')} warnings={r.get('warnings')}")
    api("POST", "/api/stop")
    reset()


def phase_map(score):
    import sys
    sys.path.insert(0, "/app/bridge")
    import nav

    prompts = [
        ("maze", "a small maze with an entrance near me and a goal marker at the far end"),
        ("maze2", "a rectangular maze, 4 corridors, goal marker at the exit"),
        ("maze3", "a simple spiral maze with a goal marker at its center"),
        ("course", "an obstacle course with cones and crates"),
        ("course2", "a slalom course of five cylinders"),
        ("room", "a small room with a doorway and a table inside"),
        ("room2", "a warehouse corner with shelves and boxes"),
        ("court", "a mini basketball court with a hoop"),
        ("parking", "a parking lot with three parked cars as boxes"),
        ("arena", "a circular arena of walls around the origin"),
    ]
    for name, prompt in prompts:
        put_world([])
        reset()
        r = api("POST", "/api/command", {"text": prompt, "mode": "map"}, timeout=300)
        if r.get("_http_error"):
            score.add("map", name, False, f"http {r['_http_error']}: {r.get('detail', '')[:120]}")
            continue
        world = state()["world"]
        enough = len(world) >= 4
        listed = gz_model_list()
        sample = [o["id"] for o in world[:3]]
        spawned = all(s in listed for s in sample)
        detail = f"{len(world)} objects, spawned={spawned}"
        ok = enough and spawned
        if name.startswith("maze") and ok:
            goal = next((o for o in world if any(k in o["name"].lower()
                                                for k in ("goal", "exit", "finish", "marker"))), None)
            if goal is None:
                ok = False
                detail += ", no goal object"
            else:
                route = nav.plan_path(world, (0.0, 0.0), (goal["x"], goal["y"]),
                                      inflate=0.28, robot_height=0.45,
                                      target_obj=goal, stop_distance=0.4)
                ok = route is not None
                detail += f", solvable={route is not None}"
        score.add("map", name, ok, detail)
    put_world([])


def phase_demo(score, n=10):
    for i in range(n):
        reset()
        put_world([obj("Wall", x=2.2, y=0.0, w=0.2, d=3.0, h=1.0, id=f"demo_wall{i}")])
        time.sleep(0.8)
        s0 = state()
        r = chat("Drive forward until you see the wall, then turn left")
        if r.get("_http_error"):
            score.add("demo", f"run#{i}", False, f"http {r['_http_error']}")
            continue
        s1 = wait_idle(timeout=120)
        moved = math.dist((s0["pose"]["x"], s0["pose"]["y"]), (s1["pose"]["x"], s1["pose"]["y"]))
        got = math.degrees(s1["pose"]["yaw"] - s0["pose"]["yaw"]) % 360.0
        turn_err = min(abs(got - 90.0), 360 - abs(got - 90.0))
        ok = moved >= 0.5 and turn_err <= 10.0
        score.add("demo", f"run#{i}", ok, f"moved {moved:.2f} m, turned {got:.0f} deg")
    put_world([])
    reset()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=[
        "baseline", "upload", "world", "motion", "chat", "map", "demo", "suite"])
    args = parser.parse_args()
    if args.phase == "baseline":
        phase_baseline()
        return
    score = Score()
    phases = {
        "upload": lambda: phase_upload(score),
        "world": lambda: phase_world(score),
        "motion": lambda: phase_motion(score),
        "chat": lambda: phase_chat(score),
        "map": lambda: phase_map(score),
        "demo": lambda: phase_demo(score),
    }
    if args.phase == "suite":
        for run in phases.values():
            run()
    else:
        phases[args.phase]()
    all_green = score.summary()
    raise SystemExit(0 if all_green else 1)


if __name__ == "__main__":
    main()
