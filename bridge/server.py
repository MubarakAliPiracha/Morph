"""FastAPI surface. Every route keeps the exact shape the existing frontend expects.

The interesting part is what is NOT here: the LLM call, plan validation and world editing
are `llm.py`, `skills.py` and `world_ops.py` reused unchanged from the PyBullet product.
Only the simulator behind `session` is new.
"""

import math
import os
import re
import signal
import threading
import time
from pathlib import Path, PurePosixPath
from typing import Dict, List, Optional, Tuple

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool
from starlette.websockets import WebSocket, WebSocketDisconnect

import asyncio
import json

import llm
import nav
import skills
import world_ops
from session import Session

llm.load_env()

MODELS_DIR = Path("/sim/models")

app = FastAPI(title="NL-Robot-GZ bridge")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)
if MODELS_DIR.exists():
    app.mount("/files", StaticFiles(directory=str(MODELS_DIR)), name="model-files")

session = Session()


def ollama_available() -> bool:
    try:
        import requests

        return requests.get(llm.OLLAMA_BASE + "/api/tags", timeout=0.6).status_code == 200
    except Exception:
        return False


# ---------------------------------------------------------------- state for the LLM

MAP_DIRECTIVE = (
    "MAP BUILDING MODE. Only build or edit the world; set steps to [] and never move the robot. Build what the user describes "
    "as completely and as well as you can (up to about 70 objects; merge walls that run in a straight line into single long boxes). "
    "Unless the user says to add to or change the existing map, set world.clear=true and rebuild from scratch. Scale EVERYTHING to this "
    "robot using STATE robot.size_m and robot.scale_guide (corridor widths, wall height and thickness, obstacle sizes, distances, "
    "clear start area). For mazes guarantee a path from the start to an exit that the robot physically fits through.\nUser request: "
)


def describe_state() -> Dict:
    """Everything the AI needs about the robot and the world (compact JSON).

    Ported from the PyBullet server: the scale_guide maths is unchanged, only the two
    simulator reads (joint positions, end-effector) now come from ROS and forward
    kinematics instead of PyBullet.
    """
    r = lambda v: round(float(v), 2)  # noqa: E731
    vehicle = session.vehicle
    position, yaw = vehicle.position(), vehicle.yaw()
    live = session.state.snapshot_joints()

    robot: Dict = {
        "name": session.info.get("name"),
        "type": "wheeled",
        "pose": {"x": r(position[0]), "y": r(position[1]), "heading_deg": r(math.degrees(yaw))},
        "footprint_m": [r(2 * vehicle.hl), r(2 * vehicle.hw)],
        "wheels": len(vehicle.wheels),
        "sensors": {
            "range_sensor": "360-degree lidar-style, 8 m",
            "front_clearance_m": r(vehicle.scan()["front"]),
        },
        "end_effector": [r(c) for c in session.kin.fk(live)],
        "reach_m": r(session.kin.reach),
    }

    joints = []
    for spec in session.joints:
        name = spec["name"]
        if name in session.wheel_names or spec["type"] == "fixed":
            continue
        if spec["lower_limit"] is None or spec["upper_limit"] is None:
            continue
        now = live.get(name, 0.0)
        if spec["type"] == "prismatic":
            joints.append({"name": name, "type": "prismatic", "unit": "m",
                           "min": r(spec["lower_limit"]), "max": r(spec["upper_limit"]), "now": r(now)})
        else:
            joints.append({"name": name, "type": spec["type"], "unit": "deg",
                           "min": r(math.degrees(spec["lower_limit"])),
                           "max": r(math.degrees(spec["upper_limit"])),
                           "now": r(math.degrees(now))})
    robot["joints"] = joints

    length, width, height = 2 * vehicle.hl, 2 * vehicle.hw, vehicle.height
    robot["size_m"] = [r(length), r(width), r(height)]
    robot["scale_guide"] = {
        "unit_m": r(width),
        "min_corridor_width_m": r(max(width + 1.0, 1.8 * width)),
        "min_turnaround_space_m": r(2 * vehicle.radius * 1.2),
        "wall_height_m": r(max(1.0, 1.3 * height)),
        "wall_thickness_m": r(max(0.2, 0.1 * width)),
        "obstacle_sizes_m": {"small": r(0.5 * width), "medium": r(width), "large": r(2 * width)},
        "clear_start_radius_m": r(vehicle.radius * 1.5 + 0.5),
        "typical_distance_ahead_m": r(max(4.0, 4 * length)),
        "arena_half_extent_m": r(max(12.0, 8 * length)),
    }
    if session.gripper:
        robot["scale_guide"]["graspable_size_max_m"] = r(0.8 * session.gripper["opening"])
        robot["gripper"] = {
            "joints": session.gripper["joints"],
            "max_opening_m": r(session.gripper["opening"]),
            "fingers_reach_ahead_m": r(session.gripper["tip_ahead"]),
            "note": "use the grasp and release skills; they handle approach and jaw positions",
        }

    objects = []
    for obj in session.world_snapshot():
        objects.append({
            "name": obj["name"], "kind": obj["kind"],
            "center": [r(obj["x"]), r(obj["y"]), r(obj["z"] + obj["h"] / 2)],
            "size": [r(obj["w"]), r(obj["d"]), r(obj["h"])],
            "yaw_deg": r(obj.get("yaw", 0)), "dynamic": bool(obj.get("dynamic")),
        })
    return {"robot": robot, "objects": objects, "recent_commands": session.history[-6:]}


# ---------------------------------------------------------------- routes


@app.get("/api/health")
def health() -> Dict:
    status = llm.status()
    return {
        "ok": True,
        "ollama": ollama_available(),
        "model": status.get("model") or "",
        "robot": session.robot_info(),
    }


class SelectBody(BaseModel):
    source: str


@app.post("/api/robot/select")
def select_robot(body: SelectBody) -> Dict:
    # One robot is loaded per container (NLROBOT_ROBOT). 'default' is accepted because
    # the UI hardcodes it -- lib/sims.ts seeds every new sim with robotSource 'default'.
    info = session.robot_info()
    known = ("default", "rover", "nlbot", "tb3", session.info["source"])
    if body.source not in known:
        info["warnings"] = [f"Unknown robot '{body.source}'; kept {session.model.name}."]
    return info


MAX_UPLOAD_BYTES = 300 * 1024 * 1024
ACTIVE_ROBOT_FILE = Path("/sim/models/.active_robot")
UPLOAD_ID_RE = re.compile(r"^upload_[0-9a-f]{8}$")


def safe_relative_path(name: str) -> Optional[Path]:
    """A browser filename or zip entry as a path that cannot escape the upload dir."""
    parts = [p for p in PurePosixPath(name.replace("\\", "/")).parts
             if p not in ("", ".", "/")]
    if not parts or any(p == ".." for p in parts) or parts[0].endswith(":"):
        return None
    return Path(*parts)


def extract_zip(data: bytes, dest: Path) -> None:
    import io
    import zipfile

    try:
        archive = zipfile.ZipFile(io.BytesIO(data))
    except zipfile.BadZipFile:
        raise HTTPException(422, "That .zip file could not be read.")
    total = 0
    with archive:
        for member in archive.infolist():
            if member.is_dir():
                continue
            rel = safe_relative_path(member.filename)
            if rel is None:
                continue
            total += member.file_size
            if total > MAX_UPLOAD_BYTES:
                raise HTTPException(422, "Archive too large (300 MB limit).")
            target = dest / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(member) as src:
                target.write_bytes(src.read())


def rewrite_unresolvable_finds(xacro_text: str, upload_root: Path) -> str:
    """Point $(find <pkg>) somewhere useful when <pkg> is not an installed package.

    Exported URDFs reference their own source package, which is never installed in this
    container; installed packages (e.g. turtlebot3_description) still resolve normally.
    A folder in the upload named like the package is the package, so prefer it.
    ament import stays local so the module keeps importing outside the ROS environment.
    """
    from ament_index_python.packages import PackageNotFoundError, get_package_share_directory

    def substitute(match: "re.Match[str]") -> str:
        pkg = match.group(1)
        try:
            get_package_share_directory(pkg)
            return match.group(0)
        except PackageNotFoundError:
            nested = upload_root / pkg
            return str(nested if nested.is_dir() else upload_root)

    return re.sub(r"\$\(find\s+([^)\s]+)\s*\)", substitute, xacro_text)


def best_robot_description(raw_dir: Path) -> Tuple[Optional[Path], List[str]]:
    """Convert every candidate and keep the one that declares the most links.

    A zip or folder upload holds macro/include xacros alongside the real robot; the
    include-only files convert fine but produce zero links, so link count -- not file
    order -- is what identifies the actual robot. Returns (urdf_path, notes)."""
    import subprocess
    import xml.etree.ElementTree as ET

    for xacro_file in raw_dir.rglob("*.xacro"):
        text = xacro_file.read_text(encoding="utf-8", errors="replace")
        rewritten = rewrite_unresolvable_finds(text, raw_dir)
        if rewritten != text:  # in place: included files need the rewrite too
            xacro_file.write_text(rewritten)

    best, best_links, notes = None, 0, []
    for path in sorted(raw_dir.rglob("*")):
        if path.suffix.lower() not in (".urdf", ".xacro"):
            continue
        if path.name.endswith(".converted.urdf"):
            continue
        if path.suffix.lower() == ".xacro":
            done = subprocess.run(["xacro", str(path)], capture_output=True, text=True)
            if done.returncode != 0:
                notes.append(f"{path.name}: xacro failed: {done.stderr.strip()[:200]}")
                continue
            candidate = path.with_name(path.stem + ".converted.urdf")
            candidate.write_text(done.stdout)
        else:
            candidate = path
        try:
            links = len(ET.parse(candidate).getroot().findall("link"))
        except ET.ParseError as exc:
            notes.append(f"{path.name}: not valid XML: {exc}")
            continue
        if links > best_links:
            best, best_links = candidate, links
    return best, notes


@app.post("/api/robot/upload")
async def upload_robot(files: List[UploadFile] = File(...)) -> Dict:
    """Run the uploaded robot through the same converter that imported the TurtleBot3.

    Accepts loose files, a folder, or a .zip of either. The pipeline strips
    Gazebo-Classic plugins, retargets ros2_control to the sim, converts sensors to
    Fortress naming, and reports every sensor it found. The response carries
    `upload_id`; POST it to /api/robot/activate to reboot the sim into this robot.
    """
    import sys
    import uuid

    sys.path.insert(0, "/sim/tools")
    upload_id = "upload_" + uuid.uuid4().hex[:8]
    raw_dir = Path("/sim/models") / upload_id / "_raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    received = 0
    for f in files:
        rel = safe_relative_path(f.filename or "file")
        if rel is None:
            continue
        data = await f.read()
        received += len(data)
        if received > MAX_UPLOAD_BYTES:
            raise HTTPException(422, "Upload too large (300 MB limit).")
        if rel.suffix.lower() == ".zip":
            extract_zip(data, raw_dir)
            continue
        dest = raw_dir / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(data)

    urdf_path, notes = best_robot_description(raw_dir)
    if urdf_path is None:
        detail = "No usable robot description in the upload."
        raise HTTPException(422, " ".join([detail] + notes[:3]))

    try:
        import import_robot as importer
        import io
        from contextlib import redirect_stdout
        report = io.StringIO()
        with redirect_stdout(report):
            sys.argv = ["import_robot", str(urdf_path),
                        f"/sim/models/{upload_id}", upload_id, str(raw_dir)]
            importer.main()
    except Exception as exc:
        raise HTTPException(422, f"Robot conversion failed: {exc}")

    import robot_model
    uploaded = robot_model.load(f"/sim/models/{upload_id}/{upload_id}.urdf")
    sensor_names = [f"{s['type']} on {s['link']}" for s in uploaded.sensors] or ["none declared"]

    info = session.robot_info()
    info["upload_id"] = upload_id
    info["warnings"] = [
        f"Converted for Gazebo Fortress as '{upload_id}': "
        f"{len(uploaded.movable)} movable joints, wheels {uploaded.wheels or 'none'}, "
        f"sensors: {', '.join(sensor_names)}."
    ] + notes
    return info


class ActivateBody(BaseModel):
    source: str


@app.post("/api/robot/activate")
def activate_robot(body: ActivateBody) -> Dict:
    """Reboot the simulator with the chosen robot.

    The controller stack is instantiated at container boot, so switching robots means
    restarting the container: record the selection where the entrypoint reads it,
    answer this request, then signal PID 1. docker-compose `restart: unless-stopped`
    brings the container straight back up; a broken selection falls back to tb3 there.
    """
    source = body.source.strip()
    if source != "tb3" and not UPLOAD_ID_RE.match(source):
        raise HTTPException(422, f"Unknown robot '{source}'.")
    if source != "tb3" and not Path(f"/sim/models/{source}/{source}.urdf").is_file():
        raise HTTPException(422, f"No converted model named '{source}' on this sim.")
    ACTIVE_ROBOT_FILE.write_text(source)

    def reboot() -> None:
        time.sleep(1.0)  # let the HTTP response leave first
        os.kill(1, signal.SIGTERM)

    threading.Thread(target=reboot, daemon=True, name="robot-activate-reboot").start()
    return {"ok": True, "source": source, "rebooting": True}


@app.post("/api/robot/reset")
def reset_robot() -> Dict:
    return session.reset()


class WorldBody(BaseModel):
    objects: List[Dict]


@app.put("/api/world")
def put_world(body: WorldBody) -> Dict:
    session.set_world(body.objects)
    return {"ok": True}


def unreachable_warnings(actions: List[Dict]) -> List[str]:
    """Flag go_to goals no traversable route reaches, before the robot even moves.

    This is how an LLM-built maze with no entrance surfaces as an honest sentence in
    chat instead of a robot silently nosing a wall for ninety seconds.
    """
    out: List[str] = []
    world = session.scan_world()
    x, y, _ = session.state.base_2d()
    for action in actions:
        if action["primitive"] != "go_to":
            continue
        params = action["params"]
        target_obj = next((o for o in world if o["id"] == params.get("target_id")), None)
        route = nav.plan_path(
            world, (x, y), (params["x"], params["y"]),
            session.executor.inflate, session.vehicle.height,
            target_obj=target_obj, stop_distance=params.get("stop_distance", 0.0),
        )
        if route is None:
            out.append(
                f"No traversable route to ({params['x']:.1f}, {params['y']:.1f}) exists in "
                "this world - the robot will get as close as it can. Check for sealed walls."
            )
        else:
            x, y = route[-1]  # later steps start from where this one ends
    return out


class CommandBody(BaseModel):
    text: str
    mode: str = "robot"


@app.post("/api/command")
def command(body: CommandBody) -> Dict:
    text = body.text.strip()
    if not text:
        raise HTTPException(400, "Empty command")
    map_mode = body.mode == "map"

    warnings: List[str] = []
    actions: List[Dict] = []
    reply = source = None
    repeat = world_changed = False
    llm_error = None

    result = None
    try:
        result = llm.interpret(
            describe_state(),
            MAP_DIRECTIVE + text if map_mode else text,
            effort=os.environ.get("NL_ROBOT_MAP_EFFORT", "low") if map_mode else None,
        )
    except llm.LLMError as exc:
        llm_error = str(exc)

    if result is None:
        detail = "I couldn't work out what to do with that."
        if llm_error:
            detail += f" ({llm_error})"
        if not llm.status().get("provider"):
            detail += " Set ANTHROPIC_API_KEY so the planner can run."
        raise HTTPException(422, detail)

    plan, source = result["plan"], result["provider"]
    ops = plan.get("world")
    if isinstance(ops, dict) and any(ops.get(k) for k in ("clear", "remove", "add", "update")):
        new_world, ops_warnings = world_ops.apply_ops(session.world_snapshot(), ops)
        warnings += ops_warnings
        warnings += session.set_world(new_world)
        world_changed = True

    actions, skill_warnings = skills.normalize(
        plan.get("steps") or [],
        True,
        session.joints,
        session.scan_world(),  # live poses: a pushed crate is where physics left it
        session.wheel_names,
        has_gripper=bool(session.gripper),
    )
    warnings += skill_warnings
    warnings += unreachable_warnings(actions)
    reply = str(plan.get("reply") or "").strip() or None
    repeat = bool(plan.get("repeat"))
    if map_mode:
        actions, repeat = [], False

    if not actions and not world_changed:
        raise HTTPException(
            422,
            (reply + " " if reply else "") + "Nothing runnable came out of that."
            + (" " + " ".join(warnings) if warnings else ""),
        )

    path = session.vehicle.preview(actions) if actions and not map_mode else []
    if actions:
        session.submit(actions, repeat)
    if not map_mode:
        session.history.append(
            {"user": text, "reply": reply, "steps": [a["primitive"] for a in actions]}
        )

    return {
        "ok": True, "source": source, "reply": reply, "plan": actions, "repeat": repeat,
        "world": session.world_snapshot() if world_changed else None,
        "warnings": warnings, "llm": llm.status(), "llm_error": llm_error, "path": path,
    }


class PlanBody(BaseModel):
    steps: List[Dict]
    repeat: bool = False


@app.post("/api/plan")
def plan_direct(body: PlanBody) -> Dict:
    """Submit validated steps directly, skipping the LLM.

    Same validation and execution path as /api/command; exists so tests and scripts can
    exercise the executor deterministically.
    """
    actions, warnings = skills.normalize(
        body.steps,
        True,
        session.joints,
        session.scan_world(),
        session.wheel_names,
        has_gripper=bool(session.gripper),
    )
    if not actions:
        raise HTTPException(422, "Nothing runnable came out of that. " + " ".join(warnings))
    session.submit(actions, body.repeat)
    return {"ok": True, "plan": actions, "warnings": warnings,
            "path": session.vehicle.preview(actions)}


@app.get("/api/state")
def get_state() -> Dict:
    """Live pose, executor status and world readback for tests and scripts.

    `pose` is the odometry frame every controller runs on; `pose_truth` is Gazebo's
    ground-truth model pose when available -- the difference between them is odometry
    drift, worth watching whenever the robot touches something.
    """
    queued, active = session.executor.status()
    x, y, yaw = session.state.base_2d()
    ox, oy, oyaw = session.state.odom_2d()
    return {
        "pose": {"x": round(x, 4), "y": round(y, 4), "yaw": round(yaw, 4)},
        "pose_odom": {"x": round(ox, 4), "y": round(oy, 4), "yaw": round(oyaw, 4)},
        "drift": round(math.dist((x, y), (ox, oy)), 3),
        "queued": queued,
        "active": active,
        "front": session.sensor_block()["front"],
        "held": session.executor.held_id,
        "world": session.scan_world(),
    }


@app.post("/api/stop")
def stop() -> Dict:
    session.stop()
    return {"ok": True}


@app.websocket("/ws")
async def ws_state(websocket: WebSocket) -> None:
    await websocket.accept()
    try:
        while True:
            frame = await run_in_threadpool(session.snapshot)
            await websocket.send_text(json.dumps(frame))
            await asyncio.sleep(1 / 30)
    except (WebSocketDisconnect, RuntimeError):
        pass
    except Exception:
        pass
