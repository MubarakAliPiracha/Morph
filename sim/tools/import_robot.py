"""Import a stock ROS robot description and make it run on Gazebo Fortress.

Most published robot descriptions (TurtleBot3, many others) still ship Gazebo *Classic*
plugins and a real-hardware ros2_control block. Neither works under Fortress. This script
rewrites a converted URDF into something the sim can actually run, and emits a second copy
the web viewport can load.

What it changes:
  Classic libgazebo_ros_diff_drive.so   -> ignition-gazebo-diff-drive-system
  Classic libgazebo_ros2_control.so     -> gz_ros2_control-system
  real-hardware ros2_control plugin     -> gz_ros2_control/GazeboSimSystem
  sensor type="ray"                     -> type="gpu_lidar"   (Fortress naming)
  package:// mesh URIs                  -> paths each consumer can resolve

Two outputs, because the sim and the browser need different mesh URIs:
  <out>/<name>.urdf      absolute file:// meshes, for Gazebo
  <out>/<name>.web.urdf  relative meshes/, for the Three.js loader over HTTP

Usage:
  python3 import_robot.py <input.urdf> <output_dir> <robot_name> [mesh_search_root ...]
"""

import re
import shutil
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

SHARE = Path("/opt/ros/humble/share")

CLASSIC_PLUGIN_MARKERS = (
    "libgazebo_ros_diff_drive.so",
    "libgazebo_ros2_control.so",
    "libgazebo_ros_imu_sensor.so",
    "libgazebo_ros_ray_sensor.so",
    "libgazebo_ros_camera.so",
    "libgazebo_ros_joint_state_publisher.so",
)


def resolve_package_uri(uri: str) -> Path | None:
    """package://pkg/rest -> /opt/ros/humble/share/pkg/rest"""
    match = re.match(r"package://([^/]+)/(.+)", uri)
    if not match:
        return None
    candidate = SHARE / match.group(1) / match.group(2)
    return candidate if candidate.exists() else None


def collect_meshes(root: ET.Element, mesh_dir: Path, search_root: Path | None = None) -> dict:
    """Copy every referenced mesh next to the model; return uri -> filename.

    Uploaded robots arrive as a flat file set, so their mesh URIs never resolve as
    package paths; fall back to a basename search next to the source URDF.
    """
    mapping = {}
    mesh_dir.mkdir(parents=True, exist_ok=True)
    for mesh in root.iter("mesh"):
        uri = mesh.get("filename") or ""
        if not uri.lower().endswith((".stl", ".dae", ".obj")):
            continue
        source = resolve_package_uri(uri)
        if source is None and search_root is not None:
            source = next(search_root.rglob(Path(uri).name), None)
        if source is None:
            print(f"  ! could not resolve {uri}")
            continue
        target = mesh_dir / source.name
        if not target.exists():
            shutil.copy(source, target)
        mapping[uri] = source.name
    return mapping


def strip_classic_gazebo(root: ET.Element) -> int:
    """Remove Gazebo-Classic plugin blocks; they are inert or harmful under Fortress."""
    removed = 0
    for block in list(root.findall("gazebo")):
        for plugin in list(block.findall("plugin")):
            if (plugin.get("filename") or "") in CLASSIC_PLUGIN_MARKERS:
                block.remove(plugin)
                removed += 1
        # Drop the wrapper only if nothing meaningful is left in it.
        if not list(block):
            root.remove(block)
    return removed


def convert_sensors(root: ET.Element) -> list:
    """Fortress calls it gpu_lidar, not ray; give every sensor an explicit gz topic."""
    converted = []
    for block in root.findall("gazebo"):
        reference = block.get("reference") or "?"
        for sensor in block.findall("sensor"):
            kind = sensor.get("type")
            if kind in ("ray", "gpu_ray"):
                sensor.set("type", "gpu_lidar")
                kind = "gpu_lidar"
            for stale in sensor.findall("plugin"):
                sensor.remove(stale)
            for stale in sensor.findall("topic"):
                sensor.remove(stale)
            topic = ET.SubElement(sensor, "topic")
            topic.text = {"gpu_lidar": "scan", "imu": "imu", "camera": "camera"}.get(kind, kind)
            always = ET.SubElement(sensor, "always_on")
            always.text = "1"
            converted.append((reference, sensor.get("name"), kind))
    return converted


def mimic_joints(root: ET.Element) -> set:
    """Joints driven by another joint. gz_ros2_control renames these `<name>_mimic` and
    drives them itself, so commanding them is both wrong and breaks controller activation
    (the claim fails, start interfaces come back empty, the switch is rejected)."""
    return {j.get("name") for j in root.findall("joint") if j.find("mimic") is not None}


def retarget_ros2_control(root: ET.Element, wheels: set) -> list:
    """Point ros2_control at the Gazebo system, and hand the wheels to DiffDrive."""
    mimics = mimic_joints(root)
    controlled = []
    for control in root.findall("ros2_control"):
        for hardware in control.findall("hardware"):
            for plugin in hardware.findall("plugin"):
                plugin.text = "gz_ros2_control/GazeboSimSystem"
            for params in hardware.findall("param"):
                hardware.remove(params)
        for joint in list(control.findall("joint")):
            name = joint.get("name")
            if name in wheels:
                control.remove(joint)  # DiffDrive owns these
                continue
            if name in mimics:
                continue  # keep the interface, but never put it in the controller list
            for interface in list(joint.findall("command_interface")):
                joint.remove(interface)
            ET.SubElement(joint, "command_interface", {"name": "position"})
            for interface in list(joint.findall("state_interface")):
                joint.remove(interface)
            ET.SubElement(joint, "state_interface", {"name": "position"})
            ET.SubElement(joint, "state_interface", {"name": "velocity"})
            controlled.append(name)
    return controlled


def add_fortress_systems(root: ET.Element, name: str, wheels: list, separation: float, radius: float) -> None:
    block = ET.SubElement(root, "gazebo")
    diff = ET.SubElement(block, "plugin", {
        "filename": "ignition-gazebo-diff-drive-system",
        "name": "ignition::gazebo::systems::DiffDrive",
    })
    for tag, value in (
        ("left_joint", wheels[0]), ("right_joint", wheels[1]),
        ("wheel_separation", str(separation)), ("wheel_radius", str(radius)),
        ("odom_publish_frequency", "50"),
        ("topic", f"/model/{name}/cmd_vel"), ("odom_topic", f"/model/{name}/odometry"),
        ("frame_id", "odom"), ("child_frame_id", "base_link"),
    ):
        ET.SubElement(diff, tag).text = value

    control_block = ET.SubElement(root, "gazebo")
    control = ET.SubElement(control_block, "plugin", {
        "filename": "gz_ros2_control-system",
        "name": "gz_ros2_control::GazeboSimROS2ControlPlugin",
    })
    ET.SubElement(control, "parameters").text = f"/sim/config/{name}_controllers.yaml"


def write_controllers_yaml(path: Path, joints: list) -> None:
    listing = "\n".join(f"      - {j}" for j in joints)
    path.write_text(
        "# Generated by import_robot.py. One JointTrajectoryController spans the arm and\n"
        "# gripper so a single FollowJointTrajectory client serves every arm skill.\n"
        "controller_manager:\n"
        "  ros__parameters:\n"
        "    update_rate: 100\n"
        "    joint_state_broadcaster:\n"
        "      type: joint_state_broadcaster/JointStateBroadcaster\n"
        "    arm_controller:\n"
        "      type: joint_trajectory_controller/JointTrajectoryController\n"
        "\n"
        "arm_controller:\n"
        "  ros__parameters:\n"
        "    joints:\n" + listing + "\n"
        "    command_interfaces: [position]\n"
        "    state_interfaces: [position, velocity]\n"
        "    allow_partial_joints_goal: true\n"
        "    state_publish_rate: 50.0\n"
        "    action_monitor_rate: 20.0\n"
        "    constraints:\n"
        "      stopped_velocity_tolerance: 0.05\n"
        "      goal_time: 0.0\n",
        encoding="utf-8",
    )


def main() -> None:
    source, out_dir, name = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
    out_dir.mkdir(parents=True, exist_ok=True)
    tree = ET.parse(source)
    root = tree.getroot()
    root.set("name", name)

    mesh_map = collect_meshes(root, out_dir / "meshes", source.parent)
    print(f"  meshes copied      : {len(mesh_map)}")

    removed = strip_classic_gazebo(root)
    print(f"  classic plugins cut: {removed}")

    sensors = convert_sensors(root)
    for reference, sensor_name, kind in sensors:
        print(f"  sensor kept        : {sensor_name} ({kind}) on {reference}")

    wheels = [j.get("name") for j in root.findall("joint")
              if j.get("type") == "continuous" and "wheel" in (j.get("name") or "")]
    controlled = retarget_ros2_control(root, set(wheels))
    print(f"  ros2_control joints: {controlled}")

    # Wheel geometry, read off the URDF rather than hardcoded.
    separation, radius = 0.287, 0.033
    origins = []
    for joint in root.findall("joint"):
        if joint.get("name") in wheels:
            origin = joint.find("origin")
            if origin is not None:
                origins.append([float(v) for v in origin.get("xyz", "0 0 0").split()])
    if len(origins) == 2:
        separation = round(abs(origins[0][1] - origins[1][1]), 4)
    for link in root.findall("link"):
        if link.get("name") in (w.replace("_joint", "_link") for w in wheels):
            cylinder = link.find(".//collision/geometry/cylinder") or link.find(".//visual/geometry/cylinder")
            if cylinder is not None:
                radius = float(cylinder.get("radius"))
    print(f"  diff drive         : separation={separation} radius={radius} wheels={wheels}")

    add_fortress_systems(root, name, wheels, separation, radius)
    write_controllers_yaml(Path("/sim/config") / f"{name}_controllers.yaml", controlled)

    # Sim copy: absolute mesh paths Gazebo can open directly.
    for mesh in root.iter("mesh"):
        uri = mesh.get("filename") or ""
        if uri in mesh_map:
            mesh.set("filename", f"file://{out_dir}/meshes/{mesh_map[uri]}")
    tree.write(out_dir / f"{name}.urdf", encoding="unicode", xml_declaration=True)

    # Web copy: relative paths the Three.js loader resolves against root_url.
    for mesh in root.iter("mesh"):
        uri = mesh.get("filename") or ""
        if uri.startswith("file://"):
            mesh.set("filename", "meshes/" + uri.rsplit("/", 1)[1])
    tree.write(out_dir / f"{name}.web.urdf", encoding="unicode", xml_declaration=True)

    print(f"  wrote              : {out_dir}/{name}.urdf  and  {name}.web.urdf")


if __name__ == "__main__":
    main()
