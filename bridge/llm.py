"""Natural language -> robot plan, using an LLM.

The model never writes code and never touches physics. It receives the robot's live state plus the world and returns
a structured plan (skills + optional world edits) through a single tool call. `skills.py` validates that plan and
the executors run it.

Providers (first available wins):
  1. Anthropic API   - set ANTHROPIC_API_KEY (env var or backend/.env). Model: NL_ROBOT_MODEL, default claude-opus-5.
  2. Ollama (local)  - if it is running on :11434. Model: NL_ROBOT_OLLAMA_MODEL, default llama3.1.
  3. None            - the caller falls back to the built-in phrase parser.
"""

import json
import os
import re
from pathlib import Path
from typing import Dict, Optional

import requests

BACKEND_DIR = Path(__file__).resolve().parent
OLLAMA_BASE = "http://127.0.0.1:11434"
DEFAULT_MODEL = "claude-opus-5"


class LLMError(RuntimeError):
    pass


def load_env() -> None:
    """Tiny .env loader (KEY=VALUE lines) so users can drop their API key in backend/.env."""
    for path in (BACKEND_DIR / ".env", BACKEND_DIR.parent / ".env"):
        if not path.is_file():
            continue
        for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


SYSTEM_PROMPT = """You are the control brain of a robot simulator. A non-technical person types what they want in everyday language. Requests are often short, vague or lazy ("avoid obstacles", "hit the box", "make a maze", "shoot a basket"). Work out what they actually mean and make it happen by calling submit_plan exactly once. Never ask questions back: make sensible assumptions and state them briefly in `reply`. If the request is very specific, follow it exactly.

# The world
- Units are metres and degrees. Z is up. The robot starts at the origin (0,0) facing +X; +Y is to its left. Positions are world coordinates.
- Each message contains STATE (JSON): the robot (type, footprint, pose, joints with limits and current angles, end-effector, reach, sensors), every object in the world (name, kind, center, size, yaw, dynamic, hit_point), and recent commands. Use it: refer to objects by their exact name, respect joint limits, and respect reach.
- Wheeled robots have a 360-degree range sensor. The skills below use it, so you never need to hand-code sensing.

# Skills (each step is {"skill": name, ...params}; steps run in order)
Wheeled robots:
- avoid_obstacles {duration?}: roam forward, steering around anything in the way, left OR right, whichever side is open. It does not collide. Use it for: avoid / dodge / wander / roam / explore / patrol / drive around / don't hit anything. It runs until the user presses Stop unless you give a duration.
- go_to {target: "<object name>", stop_distance?} or {x, y}: drive to an object or point, steering around other obstacles.
- drive {distance? (m; negative = reverse) or duration? (s), speed? (m/s, 0.1-0.6), until_front_within? (m), safe? (default true; set false only when the user wants to touch or ram something)}
- turn {angle_degrees}: positive = left, negative = right.
- face {target | x, y}: turn to look at something.
Joints on any robot (arm, gripper, lift, mast). Wheeled robots can have these too: STATE robot.joints lists every non-wheel joint. Combine them freely with drive/go_to, e.g. drive up to an object, then close the gripper, then lift.
- move_ee {position:[x,y,z]} or {target:"<object>", offset?:[dx,dy,dz]}, duration?: move the end effector there with inverse kinematics.
- set_joints {joints:{"<joint name>": value}, relative?, duration?}: value is in degrees for rotating joints and in METRES for sliding (prismatic) joints; STATE gives each joint's unit, min, max and current value. Absolute unless relative is true. Stay inside min/max.
- flap {joint_names?, amplitude_degrees?, freq?, duration?} and circle {joint_names?, amplitude_degrees?, freq?, duration?}: rhythmic motion.
Grippers (STATE robot.gripper exists): grasp {target:"<object name>"} opens the jaws, drives up so the object is between the fingers, and closes on it. release {} opens the jaws. Prefer these over hand-driving finger joints. After a grasp the object is held: drive/turn/go_to carries it. Use grasp on a movable object (see dynamic below).
Any robot: wait {duration}, stop {}.
Only use skills that fit this robot type. Set `repeat: true` to loop the whole plan until the user presses Stop.

# Building and changing the world
`world` = {clear?, remove?:[names], add?:[objects], update?:[{name, ...changed fields}]}. Applied before the steps run.
Object: {name, kind: box|cylinder|sphere|cone|pyramid|wedge, x, y, z?, w, d, h, yaw?, color?, dynamic?}.
- w, d, h are full sizes in metres (x, y, z). Cylinder/cone: w is the diameter. Sphere: w is the diameter. z is the elevation of the bottom (0 = on the floor). yaw rotates about Z in degrees. Wedge is a ramp rising toward +X.
- dynamic: true makes it a free-moving physics object (balls, crates the robot can push). Everything else is fixed.
- Walls are thin boxes: for a wall running along Y use w=0.2, d=length; along X use w=length, d=0.2.
- Shapes are solid. For a container or basket, build it from a floor slab plus four thin walls.
- SCALE EVERYTHING TO THE ROBOT. STATE robot.size_m is [length, width, height] and robot.scale_guide gives ready-made numbers derived from the robot's real size: unit_m, min_corridor_width_m, min_turnaround_space_m, wall_height_m, wall_thickness_m, obstacle_sizes_m {small, medium, large}, clear_start_radius_m, typical_distance_ahead_m, arena_half_extent_m, graspable_size_max_m (arms report reach-based sizes instead). Use these for every object you create or move. Never fall back to fixed sizes like 1 m boxes: a 1 m box is a giant to a 0.3 m robot and a pebble to a 7 m truck. Sizes in the examples below are for a small 0.4 m wide rover; multiply by robot width / 0.4 to adapt them.
- Layout: keep robot.scale_guide.clear_start_radius_m clear around the origin so the robot isn't spawned inside something, unless asked otherwise. Keep passages at least min_corridor_width_m wide, and junctions or turning bays at least min_turnaround_space_m across. Prefer 5-12 objects for "obstacle course/room/map", more only for mazes. Give objects short descriptive names. Use varied colours.
- Maps: build them like a designer. Mazes: lay out a grid of cells (cell size = robot footprint width + 1.0 m, at least 2 m), an outer boundary, interior walls that leave exactly one route (or a few) from the start cell to an exit, and merge collinear walls into long boxes. Rooms: outer walls with doorways, then furniture (tables, shelves, crates). Courts and arenas: boundary lines, hoops or goals from several pieces, sensible sizes. Make it look intentional: consistent wall height (about 1 m), coherent colours, no overlapping shapes, everything inside a sensible area in front of the robot (use arena_half_extent_m for the overall size).
- If the user asks for an environment AND a behaviour ("make a maze and drive through it"), do both in one plan: world first, then steps.
- If a target is beyond an arm's reach you may move it closer with world.update and say so.
- To grab something: make it dynamic if needed, check it fits within the gripper's max_opening_m (resize it via world.update if it doesn't), then use the grasp skill. Dynamic objects default to light (a few kg); set `mass` (kg) to make one heavier.
- Fixed objects cannot be lifted, carried or pushed. Whenever the task is to pick up, carry, push, throw or stack something, first make it movable with world.update {name, dynamic:true} (and resize it to something the gripper can actually hold if needed), then do the task. Do this without asking.

# Judgement
- Interpret intent, not wording. "Hit the box" for an arm = move the end effector to the box's hit_point, then return to a safe pose. "Go around" / "explore" = avoid_obstacles. "Come back" = go_to x=0,y=0. "Do it again" = repeat the last plan from recent commands.
- Choose sensible durations and distances when unspecified.
- If the robot cannot do something (for example it has no launcher for "shoot a basketball"), do the closest useful thing with the parts it has (build the court, position the joints, etc.) and say plainly in `reply` what it can't do. Never claim success you can't deliver.
- `reply`: at most two short, friendly sentences in plain language: what you'll do plus any assumption.

# Examples (illustrative; always use real names and numbers from STATE)
Wheeled, "avoid obstacles":
{"reply":"Roaming around and steering clear of everything in the way.","steps":[{"skill":"avoid_obstacles"}]}
Wheeled, "go to the red cone then come back":
{"reply":"Driving to the red cone, then back to where I started.","steps":[{"skill":"go_to","target":"Red cone","stop_distance":0.5},{"skill":"go_to","x":0,"y":0}]}
Any, "make me an obstacle course":
{"reply":"Built a course with walls, crates and a ramp ahead of the robot.","world":{"add":[{"name":"Wall A","kind":"box","x":3,"y":-1.2,"w":0.2,"d":2.5,"h":0.8,"color":"orange"},{"name":"Wall B","kind":"box","x":5.5,"y":1.2,"w":0.2,"d":2.5,"h":0.8,"color":"blue"},{"name":"Crate","kind":"box","x":4,"y":0.4,"w":0.5,"d":0.5,"h":0.5,"dynamic":true,"color":"brown"},{"name":"Ramp","kind":"wedge","x":7.5,"y":0,"w":1.5,"d":1.2,"h":0.4,"color":"green"}]},"steps":[]}
Arm, "hit the box once" (box hit_point [0.62,0.1,0.35], home end-effector [0.0,0.0,1.2]):
{"reply":"Swinging the arm to the box and back.","steps":[{"skill":"move_ee","position":[0.62,0.1,0.35],"duration":2},{"skill":"move_ee","position":[0.0,0.0,1.2],"duration":2}]}
"""


PLAN_SCHEMA: Dict = {
    "type": "object",
    "properties": {
        "reply": {"type": "string", "description": "One or two short friendly sentences telling the user what you will do."},
        "world": {
            "type": "object",
            "description": "Optional edits to the world, applied before the steps.",
            "properties": {
                "clear": {"type": "boolean"},
                "remove": {"type": "array", "items": {"type": "string"}},
                "add": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "kind": {"type": "string", "enum": ["box", "cylinder", "sphere", "cone", "pyramid", "wedge"]},
                            "x": {"type": "number"}, "y": {"type": "number"}, "z": {"type": "number"},
                            "w": {"type": "number"}, "d": {"type": "number"}, "h": {"type": "number"},
                            "yaw": {"type": "number"},
                            "color": {"type": "string"},
                            "dynamic": {"type": "boolean"},
                            "mass": {"type": "number"},
                        },
                        "required": ["kind", "x", "y", "w", "d", "h"],
                    },
                },
                "update": {"type": "array", "items": {"type": "object", "properties": {"name": {"type": "string"}}, "required": ["name"]}},
            },
        },
        "steps": {
            "type": "array",
            "description": "Robot actions in order. Each has a 'skill' plus that skill's parameters.",
            "items": {"type": "object", "properties": {"skill": {"type": "string"}}, "required": ["skill"]},
        },
        "repeat": {"type": "boolean", "description": "Loop the steps until the user presses Stop."},
    },
    "required": ["reply", "steps"],
}

TOOL = {
    "name": "submit_plan",
    "description": "Submit the robot's plan: what to say to the user, optional world edits, and the ordered steps to run.",
    "input_schema": PLAN_SCHEMA,
}


def status() -> Dict:
    if os.environ.get("ANTHROPIC_API_KEY"):
        return {"provider": "anthropic", "model": os.environ.get("NL_ROBOT_MODEL", DEFAULT_MODEL)}
    try:
        if requests.get(OLLAMA_BASE + "/api/tags", timeout=0.6).status_code == 200:
            return {"provider": "ollama", "model": os.environ.get("NL_ROBOT_OLLAMA_MODEL", "llama3.1")}
    except requests.RequestException:
        pass
    return {"provider": None, "model": None}


def _user_message(state: Dict, text: str) -> str:
    return f"STATE:\n{json.dumps(state, separators=(',', ':'))}\n\nREQUEST: {text}"


def _call_anthropic(state: Dict, text: str, effort: Optional[str] = None) -> Dict:
    import anthropic

    model = os.environ.get("NL_ROBOT_MODEL", DEFAULT_MODEL)
    headers = {}
    if os.environ.get("ANTHROPIC_WORKSPACE_ID"):
        # Needed for API keys that aren't scoped to a single workspace.
        headers["anthropic-workspace-id"] = os.environ["ANTHROPIC_WORKSPACE_ID"]
    client = anthropic.Anthropic(timeout=90.0, max_retries=5, default_headers=headers or None)  # 529 "overloaded" is transient
    kwargs: Dict = dict(
        model=model, max_tokens=12000, system=SYSTEM_PROMPT, tools=[TOOL],
        messages=[{"role": "user", "content": _user_message(state, text)}],
    )
    if not model.startswith("claude-haiku"):
        kwargs["output_config"] = {"effort": effort or os.environ.get("NL_ROBOT_EFFORT", "medium")}
    try:
        try:
            # Server-side fallback: if a safety classifier declines, the API retries on another model.
            resp = client.beta.messages.create(betas=["server-side-fallback-2026-07-01"], fallbacks="default", **kwargs)
        except anthropic.BadRequestError:
            resp = client.messages.create(**kwargs)
    except anthropic.AuthenticationError:
        raise LLMError("The Anthropic API key was rejected. Check ANTHROPIC_API_KEY.")
    except anthropic.APIConnectionError:
        raise LLMError("Couldn't reach the Anthropic API (network problem).")
    except anthropic.APIStatusError as e:
        raise LLMError(f"Anthropic API error ({e.status_code}): {e.message}")

    if resp.stop_reason == "refusal":
        raise LLMError("The model declined this request.")
    for block in resp.content:
        if block.type == "tool_use" and block.name == "submit_plan" and isinstance(block.input, dict):
            return block.input
    raise LLMError("The model didn't return a plan.")


def _call_ollama(state: Dict, text: str) -> Dict:
    model = os.environ.get("NL_ROBOT_OLLAMA_MODEL", "llama3.1")
    system = (
        SYSTEM_PROMPT
        + "\n\nRespond with ONLY one JSON object that matches this schema (this is the submit_plan tool input):\n"
        + json.dumps(PLAN_SCHEMA)
    )
    try:
        r = requests.post(
            OLLAMA_BASE + "/api/chat",
            json={
                "model": model, "stream": False, "format": "json", "options": {"temperature": 0.2},
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": _user_message(state, text)}],
            },
            timeout=180,
        )
        r.raise_for_status()
        content = r.json()["message"]["content"]
    except (requests.RequestException, KeyError, ValueError) as e:
        raise LLMError(f"Ollama request failed: {e}")
    try:
        obj = json.loads(content)
    except ValueError:
        m = re.search(r"\{.*\}", content, re.DOTALL)
        if not m:
            raise LLMError("Ollama didn't return JSON.")
        try:
            obj = json.loads(m.group(0))
        except ValueError:
            raise LLMError("Ollama returned malformed JSON.")
    if not isinstance(obj, dict):
        raise LLMError("Ollama returned an unexpected shape.")
    return obj


def interpret(state: Dict, text: str, effort: Optional[str] = None) -> Optional[Dict]:
    """Return {"provider", "plan"} or None when no LLM is configured. Raises LLMError if the LLM fails."""
    st = status()
    if st["provider"] == "anthropic":
        return {"provider": "anthropic", "plan": _call_anthropic(state, text, effort)}
    if st["provider"] == "ollama":
        return {"provider": "ollama", "plan": _call_ollama(state, text)}
    return None
