"""The seam.

`server.py` (routes, /api/command, describe_state, the 30 Hz /ws loop) was written against
PyBullet but only ever touches the simulator through this one object. Reimplementing the
same public surface against ROS 2 is what lets everything above it be reused unchanged.

Public surface kept identical to the original:
    world, set_world(), joints, vehicle(.wheels/.preview), gripper,
    submit(plan, repeat), history, snapshot(), stop(), load()
"""

import math
import subprocess
import threading
import time
from typing import Dict, List, Optional

import os

import robot_model
import scan as scan_mod
import sdf as sdf_mod
from kinematics import Kinematics
from ros_node import LatestState, SimNode, start_ros
from executor import PlanExecutor

ROBOT_NAME = os.environ.get("NLROBOT_ROBOT", "tb3")
URDF_PATH = f"/sim/models/{ROBOT_NAME}/{ROBOT_NAME}.urdf"


class Vehicle:
    """The wheeled-base facade `server.py` expects (`session.vehicle`)."""

    def __init__(self, session: "Session") -> None:
        self._session = session
        self.wheels = [{"name": n} for n in session.model.wheels]
        length, width, height = session.model.footprint
        self.hl = length / 2
        self.hw = width / 2
        self.height = height
        self.radius = math.hypot(self.hl, self.hw)

    def position(self):
        x, y, _ = self._session.node.state.base_2d()
        return (x, y, 0.0)

    def yaw(self) -> float:
        return self._session.node.state.base_2d()[2]

    def scan(self) -> Dict:
        return self._session.sensor_block()

    def preview(self, actions: List[Dict]) -> List:
        """Straight-line route preview for CommandResult.path -- drawn by the viewport."""
        x, y, _ = self._session.node.state.base_2d()
        path = [[round(x, 3), round(y, 3)]]
        for action in actions:
            if action.get("primitive") in ("go_to", "grasp"):
                params = action.get("params") or {}
                if "x" in params and "y" in params:
                    path.append([round(float(params["x"]), 3), round(float(params["y"]), 3)])
        return path if len(path) > 1 else []


class Session:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.world: List[Dict] = []
        self.history: List[Dict] = []

        self.model = robot_model.load(URDF_PATH)
        self.kin = Kinematics(self.model)
        self.wheel_names = set(self.model.wheels)

        self.joints = self.model.joints
        self.gripper = {
            "joints": self.model.gripper_joints,
            "opening": self.model.max_opening,
            "tip_ahead": self.kin.arm_base[0] + self.kin.reach,
        } if self.model.gripper_joints else None
        self.info: Dict = {
            "source": ROBOT_NAME,
            "name": f"{self.model.name} (diff-drive + arm)",
            # The .web.urdf variant carries relative mesh paths the Three.js loader can
            # resolve over HTTP; the sim copy uses absolute file:// paths for Gazebo.
            "urdf_url": f"/files/{ROBOT_NAME}/{ROBOT_NAME}.web.urdf",
            "root_url": f"/files/{ROBOT_NAME}",
        }
        self.arm_info = {"reach": self.kin.reach, "ee_position": list(self.kin.fk(self.kin.home))}
        self.robot_id = 1  # server.py only ever checks this for None

        self.state = LatestState()
        self.node: SimNode = start_ros(self.state)
        self.vehicle = Vehicle(self)
        self.executor = PlanExecutor(self.node, lambda: self.scan_world(), self.kin,
                                     radius=self.vehicle.radius, height=self.vehicle.height)

        # Fallback joint pose. If gz_ros2_control fails to load, /joint_states never
        # publishes and the viewport would draw the arm in T-pose through the floor --
        # which reads as a viewport bug rather than a missing controller.
        self._default_joints = {**self.kin.home, **self.kin.gripper(self.kin.gripper_open)}
        self._spawned: Dict[str, Dict] = {}
        self._collect_strays()  # a previous backend run may have left models behind
        threading.Thread(target=self._pump, daemon=True, name="session-pump").start()

    # ---------- world ----------

    def _collect_strays(self) -> None:
        """Delete every Gazebo model this session doesn't own.

        `_spawned` lives in memory, so a backend restart orphans whatever was in the
        world: physically present, absent from every scan and viewport. An invisible
        wall is the single worst accuracy bug this product can have.
        """
        try:
            done = subprocess.run(
                ["bash", "-lc", "source /opt/ros/humble/setup.bash && ign model --list"],
                capture_output=True, text=True, timeout=20,
            )
            models = [line.strip()[2:].strip() for line in done.stdout.splitlines()
                      if line.strip().startswith("- ")]
        except (OSError, subprocess.SubprocessError) as exc:
            self.node.get_logger().warn(f"stray sweep failed: {exc}")
            return
        keep = {"ground_plane", ROBOT_NAME, *self._spawned}
        for name in models:
            if name not in keep:
                self.node.get_logger().warn(f"removing stray model '{name}'")
                self.node.delete(name)

    def world_snapshot(self) -> List[Dict]:
        with self.lock:
            return [dict(o) for o in self.world]

    def set_world(self, objects: List[Dict]) -> List[str]:
        """Reconcile Gazebo against the authored list: delete, spawn, move."""
        with self.lock:
            self.world = [dict(o) for o in objects]
        wanted = {o["id"]: o for o in self.world_snapshot()}

        for obj_id in list(self._spawned):
            if obj_id not in wanted:
                self.node.delete(obj_id)
                self._spawned.pop(obj_id, None)

        for obj_id, obj in wanted.items():
            previous = self._spawned.get(obj_id)
            if previous is None:
                x, y, z, yaw = sdf_mod.pose_of(obj)
                self.node.spawn(obj_id, sdf_mod.model_sdf(obj), x, y, z, yaw)
                self._spawned[obj_id] = dict(obj)
            elif any(previous.get(k) != obj.get(k) for k in ("w", "d", "h", "kind", "dynamic", "color")):
                self.node.delete(obj_id)
                time.sleep(0.05)
                x, y, z, yaw = sdf_mod.pose_of(obj)
                self.node.spawn(obj_id, sdf_mod.model_sdf(obj), x, y, z, yaw)
                self._spawned[obj_id] = dict(obj)
            elif any(previous.get(k) != obj.get(k) for k in ("x", "y", "z", "yaw")):
                x, y, z, yaw = sdf_mod.pose_of(obj)
                self.node.set_pose(obj_id, x, y, z, yaw)
                self._spawned[obj_id] = dict(obj)

        return sdf_mod.approximation_warnings(self.world_snapshot())

    # ---------- state ----------

    def scan_world(self) -> List[Dict]:
        """World list with dynamic objects moved to their LIVE Gazebo poses.

        Without this the synthesized scan would raycast authored positions, so an object
        the robot had pushed would still be sensed where it started -- a scan that
        disagrees with physics is exactly what a testing product must not ship.
        """
        objects = self.world_snapshot()
        live = self.state.poses_for([o["id"] for o in objects if o.get("dynamic")])
        for obj in objects:
            pose = live.get(obj["id"])
            if pose:
                obj["x"], obj["y"] = pose[0], pose[1]
                obj["z"] = pose[2] - obj["h"] / 2.0  # Gazebo reports centroid; z is bottom
        return objects

    def sensor_block(self) -> Dict:
        x, y, yaw = self.state.base_2d()
        return scan_mod.snapshot_sensor(self.scan_world(), x, y, scan_mod.LASER_HEIGHT, yaw)

    def _pump(self) -> None:
        """Publish the synthesized scan and feed the executor's reactive distances."""
        while True:
            try:
                sensor = self.sensor_block()
                self.executor.set_scan(sensor["front"], sensor["dist"])
                self.node.publish_scan(sensor)
            except Exception as exc:
                # A silently frozen scan is a blind robot with no symptom -- always log.
                self.node.get_logger().warn(f"scan pump failed: {exc}")
            time.sleep(0.1)

    def snapshot(self) -> Dict:
        """The WebSocket frame. Field shapes must match the original exactly."""
        try:
            live = {
                name: value
                for name, value in self.state.snapshot_joints().items()
                if name not in self.wheel_names
            }
            queued, active = self.executor.status()
            snap: Dict = {
                "joints": {**self._default_joints, **live},
                "queued": queued,
                "active": active,
            }
            pos, quat = self.state.base_full()
            snap["base"] = {"pos": [round(c, 3) for c in pos], "quat": [round(c, 4) for c in quat]}
            snap["sensor"] = self.sensor_block()

            # Only dynamic objects, keyed by object id -- matches the original contract.
            dynamic_ids = [o["id"] for o in self.world_snapshot() if o.get("dynamic")]
            poses = self.state.poses_for(dynamic_ids)
            if poses:
                snap["objects"] = poses
            # Additive fields -- the original contract is untouched, these ride along.
            snap["telemetry"] = self.state.telemetry()
            snap["roslog"] = self.node.roslog()
            return snap
        except Exception:
            return {"joints": {}, "queued": 0, "active": False}

    # ---------- control ----------

    def stop(self) -> None:
        self.executor.stop()

    def submit(self, actions: List[Dict], repeat: bool) -> None:
        self.executor.submit(actions, repeat)

    def load(self, _urdf_path=None, info: Optional[Dict] = None) -> Dict:
        if info:
            self.info.update(info)
        return self.robot_info()

    def robot_info(self) -> Dict:
        return {
            **self.info,
            "joints": self.joints,
            "mobile": True,
            "wheels": self.model.wheels,
            "sensors": self.model.sensors,
            "scale": {"unit_m": self.model.footprint[1], "spawn_x": 1.8, "factor": 1.0},
            "warnings": [],
        }

    def reset(self) -> Dict:
        """Explicit teardown -- ControlWorld reset is unreliable for runtime-spawned models."""
        self.stop()
        for obj_id in list(self._spawned):
            self.node.delete(obj_id)
        self._spawned.clear()
        self._collect_strays()
        self.executor.held_id = None
        self.node.set_pose(ROBOT_NAME, 0.0, 0.0, 0.06, 0.0)
        time.sleep(0.3)  # let the teleport land before declaring the new odometry zero
        self.state.rebase_odometry()
        if self.node.controller_ready(timeout=0.5):
            self.node.send_trajectory(dict(self._default_joints), 1.0)
        time.sleep(0.2)
        self.set_world(self.world_snapshot())
        return self.robot_info()
