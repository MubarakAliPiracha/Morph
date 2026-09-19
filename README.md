# NL-Robot (Morph)

Tell a robot what to do in plain English. It does it in **ROS 2 Humble + Gazebo Fortress**.

Type *"drive forward until you see the wall, then turn left"* — Claude turns it into a
validated plan, the plan runs on **real ROS 2 interfaces** (`/cmd_vel`,
`FollowJointTrajectory`, `/scan`), Gazebo simulates the physics, and you watch it live
in a browser 3D viewport. Built on [NL-Robot-Sim](https://github.com/007Aurick/NL-Robot-Sim),
ported from PyBullet to the stack robotics teams actually test on.

The default robot is a **TurtleBot3 + OpenMANIPULATOR-X** imported from the official
ROS 2 packages — real meshes, lidar, IMU and camera.

---

## Prerequisites

| Tool | Version used | Notes |
|---|---|---|
| **Docker Desktop** | 29.x | must be **running** before any `docker compose` command. All ROS/Gazebo lives in one Linux container — no native ROS install needed, works on Windows/macOS/Linux |
| **Node.js** | 18+ (built with 24) | frontend only |
| **Anthropic API key** | — | workspace-scoped, **with credits** — see step 2 |
| Disk / RAM | ~6 GB / 4 GB+ | the sim image is ~4.5 GB |

No GPU required — Gazebo runs headless with software rendering; the web viewport is the display.

---

## Build & run

### 1. Clone

```bash
git clone https://github.com/MubarakAliPiracha/Morph.git
cd Morph
```

### 2. Configure the planner key

```bash
cp .env.example .env
# then edit .env and paste your key:
#   ANTHROPIC_API_KEY=sk-ant-...
```

The key must be **workspace-scoped and have credits**. Two failure modes we hit so you
don't have to:
- an **org-level** key → `"must include the anthropic-workspace-id header"` — either use a
  workspace-scoped key, or fill `ANTHROPIC_WORKSPACE_ID` in `.env`
- a key with **no credits** → `"credit balance is too low"` — add credits in
  [Console → Plans & Billing](https://console.anthropic.com)

Without a key everything still runs **except the chat** (the sim, 3D view and world
editor don't need it).

### 3. Build and start the simulator (backend)

```bash
docker compose up -d --build
```

First build takes **10–15 minutes** (ROS 2 Humble, Gazebo Fortress, ros2_control,
TurtleBot3 packages). After that it's cached. On boot the container:

1. converts the TurtleBot3 description for Gazebo Fortress (`sim/tools/import_robot.py`)
2. starts headless Gazebo with the world (`sim/worlds/nlworld.sdf`)
3. brings up `robot_state_publisher` → spawns the robot → loads the controllers
4. bridges ROS topics/services and serves the API on **http://localhost:8000**

Sanity check:

```bash
curl http://localhost:8000/api/health          # {"ok":true,...}
docker compose run --rm sim smoke              # plugin/services/version self-test
```

### 4. Start the frontend

```bash
npm install
npm run dev            # development  -> http://localhost:3000
```

For a demo, prefer the production server — it has no dev cache to corrupt and loads faster:

```bash
npm run build && npm start
```

### 5. Verify it works

Open **http://localhost:3000** → create a simulation → press **“▶ Run the demo”**.
You should see: a wall appears → Claude plans 3 steps (checklist ticks green) → the robot
drives to the wall and turns left, camera following. The **Console** tab shows live
telemetry and every ROS command as the CLI you could type yourself.

To watch the ROS side directly:

```bash
docker compose exec sim bash -lc 'source /opt/ros/humble/setup.bash && ros2 topic list'
docker compose exec sim bash -lc 'source /opt/ros/humble/setup.bash && ros2 topic echo /odom --once'
```

---

## Project layout

```
bridge/        Python backend: FastAPI + rclpy session, plan executor,
               closed-form IK, world->SDF spawning, LLM planner (Claude)
sim/           ROS assets: world SDF, controller/bridge configs, robot
               importer (converts stock URDFs for Gazebo Fortress)
docker/        sim image + boot sequence (order matters - see entrypoint.sh)
app/ components/ lib/    Next.js frontend: 3D viewport (React Three Fiber),
               chat, plan checklist, console, landing page
```

### How a command flows

```
you type English
  -> POST /api/command -> Claude returns {steps:[{skill,...}]}
  -> skills.py validates -> executor publishes /cmd_vel or sends
     FollowJointTrajectory goals -> Gazebo physics -> /odom, /joint_states,
     /scan stream back over one WebSocket -> the viewport renders it
```

Same topics and actions a physical TurtleBot3 exposes — plans that run here map onto
real hardware.

---

## Troubleshooting (each of these actually happened)

| Symptom | Cause → fix |
|---|---|
| `error during connect: ... dockerDesktopLinuxEngine` | Docker Desktop isn't running → start it, wait for the whale |
| Changed `.env` but the old key is still used | `docker compose restart` **keeps the old environment** → `docker compose up -d --force-recreate` |
| Chat: *"not scoped to a workspace"* / *"credit balance too low"* | key type / billing — see step 2 |
| Page loads but every click is dead, or CSS/pages 404 | corrupted Next dev cache → stop the dev server, `rm -rf .next`, `npm run dev` (or use `npm run build && npm start`, which can't do this) |
| Viewport black for ~10 s on first open | 8 MB of robot meshes loading — the `$ loading robot meshes` overlay is normal |
| Robot renders away from origin after Reset | fixed in code (odometry rebase) — if you ever see it, restart the container |
| Windows: `curl -F` can't read `/tmp/...` | git-bash path — use a real Windows path with `cygpath -w` |

## Honest engineering notes

- **Lidar** is Gazebo's real `gpu_lidar` ray-tracing (headless, software rendering); a
  deterministic Python raycaster (`bridge/scan.py`) is the fallback for robots without one.
- **Grasping** is a kinematic carry (close jaws → weld object to the tool at 20 Hz).
  Fortress's DetachableJoint can't attach to models spawned at runtime.
- **No Nav2/MoveIt**: `go_to` is a P-controller on ground-truth odometry and `move_ee`
  is closed-form IK — matching the original product's fidelity with far fewer moving parts.
- Arbitrary uploaded URDFs get converted and validated (`/api/robot/upload` reports
  detected joints/sensors) but need a container restart to drive.
