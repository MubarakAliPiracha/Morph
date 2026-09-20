"""Turns an AI-produced plan into validated, executable actions.

The AI is free-form; this layer is strict. It resolves object names to coordinates, fuzzy-matches joint names,
coerces numbers, drops steps the robot can't do, and reports what it dropped so the user isn't left guessing.
"""

from typing import Dict, List, Optional, Tuple



def _nonfixed_names(joint_list: List[Dict], count: int) -> List[str]:
    return [j["name"] for j in joint_list if j["type"] != "fixed"][:count]


def _flap_params(joint_list: List[Dict], duration: float) -> Dict:
    """Inlined from the PyBullet PoC -- it was the only reason this module imported it."""
    arm = [
        j["name"]
        for j in joint_list
        if any(k in j["name"].lower() for k in ("shoulder", "elbow", "wrist", "arm"))
    ]
    if not arm:
        arm = _nonfixed_names(joint_list, 2)
    return {"joint_names": arm, "amplitude_degrees": 25, "freq": 2.0, "duration": duration}


def _circle_params(joint_list: List[Dict], duration: float) -> Dict:
    return {
        "joint_names": _nonfixed_names(joint_list, 2),
        "amplitude_degrees": 15,
        "freq": 1.0,
        "duration": duration,
    }

MOBILE = {"drive", "turn", "go_to", "face", "avoid_obstacles"}
ARM = {"set_joints", "move_ee", "flap", "circle"}
COMMON = {"wait", "stop"}
GRIP = {"grasp", "release"}
ALIASES = {
    "move": "drive", "forward": "drive", "reverse": "drive", "rotate": "turn", "spin": "turn",
    "wander": "avoid_obstacles", "explore": "avoid_obstacles", "roam": "avoid_obstacles", "avoid": "avoid_obstacles",
    "patrol": "avoid_obstacles", "navigate": "go_to", "move_to": "go_to", "goto": "go_to", "approach": "go_to",
    "go": "go_to", "look_at": "face", "reach": "move_ee", "move_end_effector": "move_ee", "pose": "set_joints",
    "move_joints": "set_joints", "pause": "wait", "sleep": "wait", "halt": "stop", "wave": "flap",
    "grab": "grasp", "pick_up": "grasp", "pickup": "grasp", "pick": "grasp", "hold": "grasp", "clamp": "grasp",
    "drop": "release", "let_go": "release", "open_gripper": "release", "put_down": "release",
}


def find_object(world: List[Dict], name) -> Optional[Dict]:
    n = str(name).strip().lower()
    for o in world:
        if o["name"].lower() == n:
            return o
    for o in world:
        if n and (n in o["name"].lower() or o["name"].lower() in n):
            return o
    return None


def _f(v, default=None):
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def _match_joint(name: str, joints: List[Dict]) -> Optional[str]:
    n = str(name).strip().lower()
    names = [j["name"] for j in joints if j["type"] != "fixed"]
    for cand in names:
        if cand.lower() == n:
            return cand
    for cand in names:
        if n and (n in cand.lower() or cand.lower() in n):
            return cand
    return None


def normalize(steps: List[Dict], mobile: bool, joints: List[Dict], world: List[Dict], wheel_names=(), has_gripper: bool = False) -> Tuple[List[Dict], List[str]]:
    actions: List[Dict] = []
    warnings: List[str] = []
    # A wheeled base may also carry a gripper, lift or arm; those joints stay controllable.
    has_attachments = any(j["type"] != "fixed" and j["name"] not in wheel_names for j in joints)
    for raw in steps or []:
        if not isinstance(raw, dict):
            continue
        skill = str(raw.get("skill") or raw.get("primitive") or "").strip().lower().replace(" ", "_").replace("-", "_")
        skill = ALIASES.get(skill, skill)
        params = {k: v for k, v in raw.items() if k not in ("skill", "primitive", "params")}
        if isinstance(raw.get("params"), dict):
            params = {**raw["params"], **params}

        if skill not in MOBILE | ARM | COMMON | GRIP:
            warnings.append(f"Skipped unknown step '{skill or '?'}'.")
            continue
        if skill in MOBILE and not mobile:
            warnings.append(f"Skipped '{skill}': this robot has no drivable wheels.")
            continue
        if skill in GRIP and not has_gripper:
            warnings.append(f"Skipped '{skill}': this robot has no gripper.")
            continue
        if skill in ("set_joints", "move_ee") and mobile and not has_attachments:
            warnings.append(f"Skipped '{skill}': this robot is a wheeled base with no arm, gripper or lift joints.")
            continue

        out: Dict = {}
        if skill == "wait":
            out = {"duration": max(0.0, min(120.0, _f(params.get("duration", params.get("seconds")), 1.0)))}
        elif skill == "drive":
            for k in ("distance", "duration", "speed", "until_front_within"):
                if _f(params.get(k)) is not None:
                    out[k] = _f(params[k])
            if _f(params.get("until_within")) is not None:
                out["until_front_within"] = _f(params["until_within"])
            if "distance" in out and out["distance"] < 0:
                out["reverse"] = True
                out["distance"] = abs(out["distance"])
            if params.get("reverse"):
                out["reverse"] = True
            if params.get("safe") is False:
                out["safe"] = False
        elif skill == "turn":
            ang = _f(params.get("angle_degrees", params.get("degrees", params.get("angle"))))
            d = str(params.get("direction", "")).lower()
            if ang is None:
                ang = 90.0
            if d == "right":
                ang = -abs(ang)
            elif d == "left":
                ang = abs(ang)
            out = {"angle_degrees": ang}
        elif skill in ("go_to", "face"):
            tgt = params.get("target")
            if tgt is not None:
                obj = find_object(world, tgt)
                if obj is None:
                    warnings.append(f"Skipped '{skill}': there's no object called '{tgt}'.")
                    continue
                # target_id lets the executor track the live object and measure arrival
                # to its footprint edge instead of its (possibly far away) center.
                out = {"x": obj["x"], "y": obj["y"], "target_id": obj["id"]}
                if skill == "go_to":
                    out["stop_distance"] = _f(params.get("stop_distance"), 0.5)
            elif _f(params.get("x")) is not None and _f(params.get("y")) is not None:
                out = {"x": _f(params["x"]), "y": _f(params["y"])}
            else:
                warnings.append(f"Skipped '{skill}': no target or coordinates given.")
                continue
            for k in ("speed", "duration"):
                if _f(params.get(k)) is not None:
                    out[k] = _f(params[k])
        elif skill == "grasp":
            obj = find_object(world, params.get("target", ""))
            if obj is not None:
                out = {"x": obj["x"], "y": obj["y"], "target_id": obj["id"]}
            elif not mobile:
                out = {}
            else:
                warnings.append(f"Skipped 'grasp': there's no object called '{params.get('target')}'.")
                continue
        elif skill == "avoid_obstacles":
            for k in ("duration", "speed"):
                if _f(params.get(k)) is not None:
                    out[k] = _f(params[k])
        elif skill == "set_joints":
            mapped = {}
            for name, deg in (params.get("joints") or {}).items():
                real = _match_joint(name, joints)
                if real in wheel_names:
                    warnings.append(f"Ignored '{name}': wheels are driven with drive/turn, not set_joints.")
                    continue
                if real is None or _f(deg) is None:
                    warnings.append(f"Ignored unknown joint '{name}'.")
                    continue
                mapped[real] = _f(deg)
            if not mapped:
                warnings.append("Skipped 'set_joints': no valid joints.")
                continue
            out = {"joints": mapped, "relative": bool(params.get("relative", False)), "duration": _f(params.get("duration"), 1.5)}
        elif skill == "move_ee":
            pos = params.get("position")
            if params.get("target") is not None:
                obj = find_object(world, params["target"])
                if obj is None:
                    warnings.append(f"Skipped 'move_ee': there's no object called '{params['target']}'.")
                    continue
                off = params.get("offset") or [0, 0, 0]
                pos = [obj["x"] + off[0], obj["y"] + off[1], obj["z"] + obj["h"] / 2 + off[2]]
            if not (isinstance(pos, (list, tuple)) and len(pos) >= 3 and all(_f(c) is not None for c in pos[:3])):
                warnings.append("Skipped 'move_ee': no valid position.")
                continue
            out = {"position": [float(c) for c in pos[:3]], "duration": _f(params.get("duration"), 2.0)}
        elif skill in ("flap", "circle"):
            dur = _f(params.get("duration"), 3.0)
            names = [m for m in (_match_joint(n, joints) for n in params.get("joint_names") or []) if m]
            base = _flap_params(joints, dur) if skill == "flap" else _circle_params(joints, dur)
            if names:
                base["joint_names"] = names
            for k in ("amplitude_degrees", "freq"):
                if _f(params.get(k)) is not None:
                    base[k] = _f(params[k])
            out = base
        actions.append({"primitive": skill, "params": out})
    return actions, warnings
