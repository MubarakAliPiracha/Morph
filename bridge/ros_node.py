"""The rclpy side of the bridge: topic caching, cmd_vel, trajectory goals, Gazebo services.

Threading model (three threads, cleanly separated):

  main      uvicorn / FastAPI
  ros       MultiThreadedExecutor.spin() -- callbacks ONLY assign into LatestState
  executor  the plan worker (see executor.py), blocking and sequential

Subscriber callbacks deliberately contain no logic. Everything that needs to reason about
state does so from LatestState under its lock, which keeps the 30 Hz snapshot cheap and
makes the executor easy to follow.

Note on time: `use_sim_time` is never set anywhere. The world is locked to real_time_factor
1.0 and everything runs on wall clock. Half-propagated use_sim_time is a classic way to end
up with trajectory goals that silently never start.
"""

import math
import os
import threading
import time
from typing import Dict, List, Optional, Tuple

import rclpy
from builtin_interfaces.msg import Duration as DurationMsg
from control_msgs.action import FollowJointTrajectory
from geometry_msgs.msg import Pose, Twist
from nav_msgs.msg import Odometry
from rclpy.action import ActionClient
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy
from ros_gz_interfaces.srv import DeleteEntity, SetEntityPose, SpawnEntity
from ros_gz_interfaces.msg import Entity, EntityFactory
from sensor_msgs.msg import Imu, JointState, LaserScan
from tf2_msgs.msg import TFMessage
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

WORLD_NAME = os.environ.get("NLROBOT_WORLD", "nlworld")
# The spawned model's name (entrypoint.sh: `ros2 run ros_gz_sim create -name "$ROBOT"`).
ROBOT_MODEL = os.environ.get("NLROBOT_ROBOT", "tb3")


def quat_to_yaw(x: float, y: float, z: float, w: float) -> float:
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def yaw_to_quat(yaw: float) -> Tuple[float, float, float, float]:
    return (0.0, 0.0, math.sin(yaw / 2.0), math.cos(yaw / 2.0))


def wrap_pi(angle: float) -> float:
    return (angle + math.pi) % (2 * math.pi) - math.pi


class LatestState:
    """Everything the snapshot and the executor read, behind one lock."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.joints: Dict[str, float] = {}
        self.base_pos: List[float] = [0.0, 0.0, 0.0]
        self.base_quat: List[float] = [0.0, 0.0, 0.0, 1.0]
        self.model_poses: Dict[str, List[float]] = {}
        self.have_odom = False
        self.base_twist: List[float] = [0.0, 0.0]  # linear v, angular w from /odom
        self.imu: Dict[str, float] = {}
        self.real_scan: Dict[str, float] = {}
        # Odometry rebase. Teleporting the model (reset) does not reset the DiffDrive
        # odometry -- it is integrated wheel motion -- so without this the robot keeps
        # rendering at its pre-reset pose and every later plan drives from a stale frame.
        self._odo_zero = (0.0, 0.0, 0.0)  # x, y, yaw subtracted from every reading
        # Ground-truth pose from Gazebo. Integrated odometry drifts on wheel slip
        # (contact, tight turns) while world objects live in the WORLD frame; every
        # consumer of base pose prefers this and falls back to odometry only when the
        # pose bridge is silent.
        self._truth: Optional[Tuple[List[float], List[float]]] = None
        self._truth_at = 0.0

    def set_joints(self, names, positions) -> None:
        with self._lock:
            for name, position in zip(names, positions):
                self.joints[name] = float(position)

    def set_base(self, pos, quat, twist=None) -> None:
        with self._lock:
            x0, y0, yaw0 = self._odo_zero
            raw_yaw = quat_to_yaw(float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3]))
            dx, dy = float(pos[0]) - x0, float(pos[1]) - y0
            cos0, sin0 = math.cos(-yaw0), math.sin(-yaw0)
            self._raw_base = (float(pos[0]), float(pos[1]), raw_yaw)
            self.base_pos = [dx * cos0 - dy * sin0, dx * sin0 + dy * cos0, float(pos[2])]
            qx, qy, qz, qw = yaw_to_quat(raw_yaw - yaw0)
            self.base_quat = [qx, qy, qz, qw]
            if twist is not None:
                self.base_twist = [float(twist[0]), float(twist[1])]
            self.have_odom = True

    def set_truth(self, pos: List[float], quat: List[float]) -> None:
        with self._lock:
            self._truth = (list(pos), list(quat))
            self._truth_at = time.time()

    def _current(self) -> Tuple[List[float], List[float]]:
        """Ground truth when fresh, odometry otherwise. Callers hold the lock."""
        if self._truth is not None and time.time() - self._truth_at < 0.6:
            return self._truth
        return self.base_pos, self.base_quat

    def rebase_odometry(self) -> None:
        """Declare the robot's CURRENT odometry pose to be the origin (used by reset)."""
        with self._lock:
            self._odo_zero = getattr(self, "_raw_base", (0.0, 0.0, 0.0))
            self.base_pos = [0.0, 0.0, 0.0]
            self.base_quat = [0.0, 0.0, 0.0, 1.0]

    def set_imu(self, data: Dict[str, float]) -> None:
        with self._lock:
            self.imu = data

    def set_real_scan(self, data: Dict[str, float]) -> None:
        with self._lock:
            self.real_scan = data

    def telemetry(self) -> Dict:
        """Live readings panel: everything a user would watch during a test run."""
        with self._lock:
            x, y = self.base_pos[0], self.base_pos[1]
            yaw = quat_to_yaw(*self.base_quat)
            out: Dict = {
                "odom": {"x": round(x, 3), "y": round(y, 3),
                          "yaw_deg": round(math.degrees(yaw), 1),
                          "v": round(self.base_twist[0], 3),
                          "w": round(self.base_twist[1], 3)},
            }
            if self.imu:
                out["imu"] = self.imu
            if self.real_scan:
                out["lidar"] = self.real_scan
            return out

    def set_model_pose(self, name: str, pose: List[float]) -> None:
        with self._lock:
            self.model_poses[name] = pose

    def snapshot_joints(self) -> Dict[str, float]:
        with self._lock:
            return {k: round(v, 5) for k, v in self.joints.items()}

    def odom_2d(self) -> Tuple[float, float, float]:
        """Raw rebased odometry, kept visible so drift stays measurable."""
        with self._lock:
            return self.base_pos[0], self.base_pos[1], quat_to_yaw(*self.base_quat)

    def base_2d(self) -> Tuple[float, float, float]:
        with self._lock:
            pos, quat = self._current()
            return pos[0], pos[1], quat_to_yaw(*quat)

    def base_full(self) -> Tuple[List[float], List[float]]:
        with self._lock:
            pos, quat = self._current()
            return list(pos), list(quat)

    def poses_for(self, ids) -> Dict[str, List[float]]:
        with self._lock:
            return {i: self.model_poses[i] for i in ids if i in self.model_poses}


class SimNode(Node):
    def __init__(self, state: LatestState) -> None:
        super().__init__("nl_robot_bridge")
        self.state = state

        self.cmd_vel = self.create_publisher(Twist, "/cmd_vel", 10)
        self.scan_pub = self.create_publisher(LaserScan, "/scan", 10)

        self.create_subscription(JointState, "/joint_states", self._on_joints, 20)
        self.create_subscription(Odometry, "/odom", self._on_odom, 20)
        self.create_subscription(TFMessage, "/gz_poses", self._on_poses, 20)
        self.create_subscription(Imu, "/imu", self._on_imu, 10)
        self.create_subscription(LaserScan, "/gz_scan", self._on_real_scan, 10)

        # Every ROS-level command the bridge issues, as the CLI you would type by hand.
        self._log_lock = threading.Lock()
        self._roslog: List[Dict] = []
        self._last_twist_logged = (None, None)

        self.traj_client = ActionClient(self, FollowJointTrajectory, "/arm_controller/follow_joint_trajectory")
        self._goal_handle = None  # in-flight arm goal, so /api/stop can cancel it

        self.spawn_cli = self.create_client(SpawnEntity, f"/world/{WORLD_NAME}/create")
        self.delete_cli = self.create_client(DeleteEntity, f"/world/{WORLD_NAME}/remove")
        self.setpose_cli = self.create_client(SetEntityPose, f"/world/{WORLD_NAME}/set_pose")

    # ---------- subscriptions: assignment only ----------

    def _on_joints(self, msg: JointState) -> None:
        self.state.set_joints(msg.name, msg.position)

    def _on_odom(self, msg: Odometry) -> None:
        p, q = msg.pose.pose.position, msg.pose.pose.orientation
        tw = msg.twist.twist
        self.state.set_base([p.x, p.y, p.z], [q.x, q.y, q.z, q.w], (tw.linear.x, tw.angular.z))

    def _on_imu(self, msg: Imu) -> None:
        a, g = msg.linear_acceleration, msg.angular_velocity
        self.state.set_imu({"ax": round(a.x, 3), "ay": round(a.y, 3), "az": round(a.z, 3),
                            "gx": round(g.x, 3), "gy": round(g.y, 3), "gz": round(g.z, 3)})

    def _on_real_scan(self, msg: LaserScan) -> None:
        finite = [r for r in msg.ranges if math.isfinite(r)]
        n = max(1, len(msg.ranges))
        front_i = int(round((0.0 - msg.angle_min) / (msg.angle_increment or 1.0))) % n
        front = msg.ranges[front_i] if front_i < len(msg.ranges) else float("inf")
        self.state.set_real_scan({
            "front": round(front, 2) if math.isfinite(front) else None,
            "min": round(min(finite), 2) if finite else None,
            "rays": len(msg.ranges),
            "range_max": round(msg.range_max, 2),
        })

    def _on_poses(self, msg: TFMessage) -> None:
        for tf in msg.transforms:
            t, r = tf.transform.translation, tf.transform.rotation
            if tf.child_frame_id == ROBOT_MODEL:
                self.state.set_truth([t.x, t.y, t.z], [r.x, r.y, r.z, r.w])
            self.state.set_model_pose(
                tf.child_frame_id,
                [round(t.x, 3), round(t.y, 3), round(t.z, 3),
                 round(r.x, 4), round(r.y, 4), round(r.z, 4), round(r.w, 4)],
            )

    # ---------- the ROS command log ----------

    def log_command(self, cli: str, kind: str = "topic") -> None:
        with self._log_lock:
            self._roslog.append({"t": round(time.time(), 2), "kind": kind, "cmd": cli})
            del self._roslog[:-80]  # ring buffer

    def roslog(self, last: int = 40) -> List[Dict]:
        with self._log_lock:
            return list(self._roslog[-last:])

    # ---------- actuation ----------

    def drive(self, linear: float, angular: float) -> None:
        msg = Twist()
        msg.linear.x = float(linear)
        msg.angular.z = float(angular)
        self.cmd_vel.publish(msg)
        # 20 Hz publishing would flood the log; record only when the command changes.
        key = (round(float(linear), 2), round(float(angular), 2))
        if key != self._last_twist_logged:
            self._last_twist_logged = key
            self.log_command(
                f'ros2 topic pub /cmd_vel geometry_msgs/msg/Twist '
                f'"{{linear: {{x: {key[0]}}}, angular: {{z: {key[1]}}}}}"')

    def halt(self) -> None:
        self.drive(0.0, 0.0)

    def controller_ready(self, timeout: float = 2.0) -> bool:
        """Is the arm controller up? Absent is an expected degraded mode, not an error."""
        return self.traj_client.wait_for_server(timeout_sec=timeout)

    def send_trajectory(self, joints: Dict[str, float], duration: float) -> bool:
        """One-point FollowJointTrajectory goal. Returns False if the controller is absent.

        A missing controller is an expected degraded mode (the base demo does not need
        gz_ros2_control at all), so this reports rather than raises.
        """
        if not self.controller_ready():
            return False
        goal = FollowJointTrajectory.Goal()
        goal.trajectory = JointTrajectory()
        goal.trajectory.joint_names = list(joints.keys())
        point = JointTrajectoryPoint()
        point.positions = [float(v) for v in joints.values()]
        secs = max(0.2, float(duration))
        point.time_from_start = DurationMsg(sec=int(secs), nanosec=int((secs % 1.0) * 1e9))
        goal.trajectory.points = [point]
        pos = ", ".join(f"{v:.2f}" for v in joints.values())
        self.log_command(
            f'ros2 action send_goal /arm_controller/follow_joint_trajectory '
            f'control_msgs/action/FollowJointTrajectory '
            f'"{{trajectory: {{joint_names: [{", ".join(joints)}], '
            f'points: [{{positions: [{pos}], time_from_start: {{sec: {int(secs)}}}}}]}}}}"',
            kind="action")
        future = self.traj_client.send_goal_async(goal)
        future.add_done_callback(self._remember_goal)
        return True

    def _remember_goal(self, future) -> None:
        try:
            self._goal_handle = future.result()
        except Exception:
            self._goal_handle = None

    def cancel_trajectory(self) -> None:
        """Best-effort cancel of any in-flight arm goal, used by /api/stop."""
        handle, self._goal_handle = self._goal_handle, None
        if handle is not None:
            try:
                handle.cancel_goal_async()
            except Exception:
                pass

    # ---------- Gazebo entity services ----------

    def _pose_msg(self, x: float, y: float, z: float, yaw: float) -> Pose:
        pose = Pose()
        pose.position.x, pose.position.y, pose.position.z = float(x), float(y), float(z)
        qx, qy, qz, qw = yaw_to_quat(yaw)
        pose.orientation.x, pose.orientation.y = qx, qy
        pose.orientation.z, pose.orientation.w = qz, qw
        return pose

    def _call_blocking(self, client, request, timeout: float = 5.0) -> bool:
        """Wait for the service response (the spinning executor thread completes it).

        Spawn and delete MUST be ordered: fire-and-forget lets a delete overtake an
        in-flight spawn during rapid world edits, orphaning an invisible model that
        physically blocks the robot while no scan or viewport shows it.
        """
        future = client.call_async(request)
        end = time.time() + timeout
        while not future.done() and time.time() < end:
            time.sleep(0.01)
        return future.done()

    def spawn(self, name: str, sdf_xml: str, x: float, y: float, z: float, yaw: float) -> bool:
        if not self.spawn_cli.wait_for_service(timeout_sec=3.0):
            return False
        req = SpawnEntity.Request()
        factory = EntityFactory()
        factory.name = name
        factory.sdf = sdf_xml
        factory.pose = self._pose_msg(x, y, z, yaw)
        factory.allow_renaming = False
        req.entity_factory = factory
        self.log_command(f"ros2 service call /world/{WORLD_NAME}/create "
                         f"ros_gz_interfaces/srv/SpawnEntity  # model '{name}'", kind="service")
        if not self._call_blocking(self.spawn_cli, req):
            self.get_logger().warn(f"spawn of '{name}' got no response")
            return False
        return True

    def delete(self, name: str) -> bool:
        if not self.delete_cli.wait_for_service(timeout_sec=2.0):
            return False
        req = DeleteEntity.Request()
        entity = Entity()
        entity.name = name
        entity.type = Entity.MODEL
        req.entity = entity
        self.log_command(f"ros2 service call /world/{WORLD_NAME}/remove "
                         f"ros_gz_interfaces/srv/DeleteEntity  # model '{name}'", kind="service")
        if not self._call_blocking(self.delete_cli, req):
            self.get_logger().warn(f"delete of '{name}' got no response")
            return False
        return True

    def set_pose(self, name: str, x: float, y: float, z: float, yaw: float) -> bool:
        if not self.setpose_cli.service_is_ready():
            return False
        req = SetEntityPose.Request()
        entity = Entity()
        entity.name = name
        entity.type = Entity.MODEL
        req.entity = entity
        req.pose = self._pose_msg(x, y, z, yaw)
        self.setpose_cli.call_async(req)
        return True

    def publish_scan(self, sensor: Dict) -> None:
        msg = LaserScan()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "laser_link"
        msg.angle_min = float(sensor["a0"])
        msg.angle_increment = float(sensor["da"])
        msg.angle_max = float(sensor["a0"]) + float(sensor["da"]) * len(sensor["dist"])
        msg.range_min = 0.05
        msg.range_max = float(sensor["range"])
        msg.ranges = [float(r) for r in sensor["dist"]]
        self.scan_pub.publish(msg)


def start_ros(state: LatestState) -> SimNode:
    rclpy.init(args=None)
    node = SimNode(state)
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    threading.Thread(target=executor.spin, daemon=True, name="ros-executor").start()
    return node
