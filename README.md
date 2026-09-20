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

### 2. Configure the Anthropic API key

```bash
cp .env.example .env
# then edit .env and paste your key:
#   ANTHROPIC_API_KEY=sk-ant-...
```

Without a key everything still runs **except the chat** (the sim, 3D view and world
editor don't need it).

### 3. Build and start the simulator (backend)

```bash
docker compose up -d --build
```

First build takes **10–15 minutes** (ROS 2 Humble, Gazebo Fortress, ros2_control,
TurtleBot3 packages). After that it's cached. 

### 4. Start the frontend

```bash
npm install
npm run dev           
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
