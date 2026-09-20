#!/usr/bin/env bash
# Boot order matters, and so does what is allowed to block.
#
# gz_ros2_control fetches the URDF from robot_state_publisher's `robot_description`
# PARAMETER, so RSP must be up, with that parameter actually set, before the spawn.
#
# Nothing arm-related may block uvicorn: the base demo (drive, turn, go_to, world
# spawning) has zero dependency on gz_ros2_control, so a controller failure must degrade
# the arm, not take down the product.
set -e
source /opt/ros/humble/setup.bash

WORLD="${NLROBOT_WORLD:-nlworld}"

# Robot selection: env var wins, then the marker /api/robot/activate writes, then tb3.
# The marker is arbitrary file content headed into paths and sed, so it is validated
# against the exact shapes the server can write, and anything else is ignored.
ROBOT="${NLROBOT_ROBOT:-}"
ACTIVE_FILE=/sim/models/.active_robot
if [ -z "$ROBOT" ] && [ -f "$ACTIVE_FILE" ]; then
  PICK="$(tr -cd 'a-z0-9_' < "$ACTIVE_FILE")"
  case "$PICK" in
    tb3|upload_[0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f])
      ROBOT="$PICK" ;;
    *) echo "[entrypoint] ignoring invalid active robot '$PICK'" ;;
  esac
fi
ROBOT="${ROBOT:-tb3}"
URDF="/sim/models/${ROBOT}/${ROBOT}.urdf"

# A selected robot whose files vanished must fall back, not crash-loop the container.
if [ "$ROBOT" != "tb3" ] && [ ! -f "$URDF" ]; then
  echo "[entrypoint] '$ROBOT' has no URDF at $URDF - falling back to tb3"
  ROBOT=tb3
  URDF="/sim/models/tb3/tb3.urdf"
fi
echo "[entrypoint] robot: $ROBOT"

# Gazebo will not find ros2_control's system plugin without this on its search path.
export IGN_GAZEBO_SYSTEM_PLUGIN_PATH="/opt/ros/humble/lib:${IGN_GAZEBO_SYSTEM_PLUGIN_PATH}"

if [ "$1" = "smoke" ]; then
  echo "--- ros2_control system plugin .so name (gz_ vs ign_) ---"
  ls /opt/ros/humble/lib | grep -i "ros2_control-system" || echo "NONE FOUND"
  echo "--- SpawnEntity service backport present? ---"
  ros2 interface show ros_gz_interfaces/srv/SpawnEntity >/dev/null 2>&1 \
    && echo "YES - use bridged ROS services" || echo "NO - fall back to 'ign service'"
  echo "--- Fortress version ---"
  ign gazebo --versions 2>/dev/null || echo "ign gazebo not on PATH"
  echo "--- robot ---"; echo "  ROBOT=$ROBOT  URDF=$URDF"; ls -la "$URDF" 2>/dev/null || echo "  MISSING"
  exit 0
fi
if [ "$1" = "shell" ]; then exec bash; fi

# Regenerate the TurtleBot3 import if it is missing (meshes live outside the image).
if [ "$ROBOT" = "tb3" ] && [ ! -f "$URDF" ]; then
  echo "[entrypoint] importing TurtleBot3 + OpenMANIPULATOR-X for Fortress..."
  D=/opt/ros/humble/share/turtlebot3_manipulation_description
  xacro "$D/urdf/turtlebot3_manipulation.urdf.xacro" > /tmp/tb3_raw.urdf
  python3 /sim/tools/import_robot.py /tmp/tb3_raw.urdf /sim/models/tb3 tb3
fi

# The bridge maps ROS /cmd_vel and /odom onto this model's Gazebo topics.
sed -e "s|/model/[a-z0-9_]*/cmd_vel|/model/${ROBOT}/cmd_vel|" \
    -e "s|/model/[a-z0-9_]*/odometry|/model/${ROBOT}/odometry|" \
    /sim/config/bridge.yaml > /tmp/bridge.yaml

# 1. headless sim server. Fortress is `ign gazebo`, NOT `gz sim` (that's Garden+).
ign gazebo -s -r -v 2 "/sim/worlds/${WORLD}.sdf" &
sleep 6

# 2. robot_state_publisher, with robot_description as a real parameter.
#    It must be a params FILE: `-p robot_description:=<xml>` fails to parse because the
#    URDF is multi-line, and passing the path positionally uses a legacy fallback that
#    does not expose the parameter gz_ros2_control reads over its service.
python3 - "$URDF" <<'PYEOF'
import sys, yaml
urdf = open(sys.argv[1]).read()
yaml.safe_dump(
    {"robot_state_publisher": {"ros__parameters": {"robot_description": urdf}}},
    open("/tmp/rsp_params.yaml", "w"),
)
PYEOF
ros2 run robot_state_publisher robot_state_publisher \
  --ros-args --params-file /tmp/rsp_params.yaml &
sleep 4

# 3. spawn -- this is what instantiates controller_manager
ros2 run ros_gz_sim create -world "$WORLD" -file "$URDF" -name "$ROBOT" -z 0.06 || true
sleep 5

# 4. controllers, in the BACKGROUND with a bounded wait. If the plugin failed to load
#    these exit on their own and the arm is simply frozen at its home pose.
(
  ros2 run controller_manager spawner joint_state_broadcaster \
    --controller-manager /controller_manager --controller-manager-timeout 30 \
    || echo "[entrypoint] joint_state_broadcaster unavailable - arm disabled, base unaffected"
  ros2 run controller_manager spawner arm_controller \
    --controller-manager /controller_manager --controller-manager-timeout 30 \
    || echo "[entrypoint] arm_controller unavailable - arm disabled, base unaffected"
) &

# 5. bridge. Topics come from YAML; services can ONLY be given as CLI args in Humble.
ros2 run ros_gz_bridge parameter_bridge \
  --ros-args -p config_file:=/tmp/bridge.yaml -- \
  "/world/${WORLD}/create@ros_gz_interfaces/srv/SpawnEntity" \
  "/world/${WORLD}/remove@ros_gz_interfaces/srv/DeleteEntity" \
  "/world/${WORLD}/set_pose@ros_gz_interfaces/srv/SetEntityPose" &
sleep 4

# 6. the product
cd /app/bridge
export NLROBOT_ROBOT="$ROBOT"
exec uvicorn server:app --host 0.0.0.0 --port 8000
