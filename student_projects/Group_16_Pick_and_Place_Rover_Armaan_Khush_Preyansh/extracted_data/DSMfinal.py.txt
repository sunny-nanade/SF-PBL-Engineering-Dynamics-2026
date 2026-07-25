"""
Mobile base + arm — final fixed version.
Behavior: strictly follows A* pathing, drops on the green tile, and ends cleanly.
"""

import heapq
import math
import os
import time
from enum import Enum, auto

import pybullet as p
import pybullet_data

# ----------------------------
# Config
# ----------------------------
START = (-6.0, -6.0, 0.1)
GOAL  = (6.0, 6.0, 0.0)

OBSTACLE_CUBES    = [(-2.0, 2.0, 0.18), (0.0, 1.2, 0.18), (2.0, -1.0, 0.18), (4.0, 3.0, 0.18)]
CUBE_HALF_EXTENTS = (0.5, 0.5, 0.3)
CYLINDER_OBSTACLES = [(-4.0, -1.0, 0.3), (1.0, 4.0, 0.3)]
CYLINDER_HEIGHT = 1.0

SAMPLE_CUBE_INDEX = 0
SAMPLE_HALF_EXTENT = 0.10
SAMPLE_POS = (
    OBSTACLE_CUBES[SAMPLE_CUBE_INDEX][0],
    OBSTACLE_CUBES[SAMPLE_CUBE_INDEX][1],
    OBSTACLE_CUBES[SAMPLE_CUBE_INDEX][2] + CUBE_HALF_EXTENTS[2] + SAMPLE_HALF_EXTENT,
)
TIME_STEP = 1.0 / 240.0
BASE_SPEED  = 6.0
TURN_GAIN   = 2.0
WHEEL_FORCE = 950
ARM_MOUNT_Z = 0.25
DRIVE_SMOOTH    = 0.85
MAX_WHEEL_DELTA = 3.6

# Automatic path planning
ROBOT_RADIUS         = 0.48
PLANNER_MARGIN       = 0.28  # Increased to prevent corner clipping
PLANNER_GRID         = 0.35
PLANNER_BOUNDS       = (-7.2, 7.2, -7.2, 7.2)
PLANNER_DIAG_COST    = 1.41421356237
DOCK_RING_RADIUS     = 1.10
DOCK_MIN_SAMPLE_DIST = 0.85
DOCK_MAX_SAMPLE_DIST = 1.30
DOCK_CANDIDATE_COUNT = 24
PATH_REACHED_DIST    = 0.65

# Arm tuning
ARM_POS_GAIN = 0.25
ARM_VEL_GAIN = 0.65
ARM_MAX_VEL  = 1.2
ARM_FORCE    = 160
ARM_DAMPING  = [0.85] * 7
ARM_REST         = [0.0,  0.60, 0.0, -1.20, 0.0, 1.50, 0.0]
ARM_STOW_JOINTS  = [0.0,  0.50, 0.0, -1.15, 0.0, 1.40, 0.0]
ARM_CARRY_JOINTS = [0.0,  0.20, 0.0, -0.95, 0.0, 1.20, 0.0]
SIM_REALTIME = True

class Phase(Enum):
    NAV_TO_SAMPLE = auto()
    PICK          = auto()
    NAV_TO_GOAL   = auto()
    PLACE         = auto()
    DONE          = auto()

# ----------------------------
# Utilities
# ----------------------------
def _wrap_angle(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi

def sim_step(count=1):
    for _ in range(count):
        p.stepSimulation()
        if SIM_REALTIME:
            time.sleep(TIME_STEP)

# ----------------------------
# Planning helpers
# ----------------------------
def _point_inside_arena(x, y, margin=0.0):
    xmin, xmax, ymin, ymax = PLANNER_BOUNDS
    return (xmin + margin <= x <= xmax - margin) and (ymin + margin <= y <= ymax - margin)

def _point_blocked_xy(x, y, clearance):
    for ox, oy, _ in OBSTACLE_CUBES:
        if abs(x - ox) <= CUBE_HALF_EXTENTS[0] + clearance and abs(y - oy) <= CUBE_HALF_EXTENTS[1] + clearance:
            return True
    for ox, oy, radius in CYLINDER_OBSTACLES:
        if (x - ox) * (x - ox) + (y - oy) * (y - oy) <= (radius + clearance) * (radius + clearance):
            return True
    return False

def _line_clear(a, b, clearance):
    dist = math.hypot(b[0] - a[0], b[1] - a[1])
    if dist < 1e-6:
        return not _point_blocked_xy(a[0], a[1], clearance)
    steps = max(2, int(dist / (PLANNER_GRID * 0.45)))
    for i in range(steps + 1):
        t = i / steps
        x = a[0] + (b[0] - a[0]) * t
        y = a[1] + (b[1] - a[1]) * t
        if _point_blocked_xy(x, y, clearance):
            return False
    return True

def _world_to_grid(x, y):
    xmin, _, ymin, _ = PLANNER_BOUNDS
    gx = int(round((x - xmin) / PLANNER_GRID))
    gy = int(round((y - ymin) / PLANNER_GRID))
    return gx, gy

def _grid_to_world(gx, gy):
    xmin, _, ymin, _ = PLANNER_BOUNDS
    return (xmin + gx * PLANNER_GRID, ymin + gy * PLANNER_GRID)

def _grid_dims():
    xmin, xmax, ymin, ymax = PLANNER_BOUNDS
    nx = int(round((xmax - xmin) / PLANNER_GRID)) + 1
    ny = int(round((ymax - ymin) / PLANNER_GRID)) + 1
    return nx, ny

def _nearest_free_grid(seed, clearance, max_radius=18):
    nx, ny = _grid_dims()
    sx, sy = seed
    def valid(g):
        return 0 <= g[0] < nx and 0 <= g[1] < ny

    if valid(seed):
        wx, wy = _grid_to_world(sx, sy)
        if _point_inside_arena(wx, wy, margin=clearance) and not _point_blocked_xy(wx, wy, clearance):
            return seed

    for r in range(1, max_radius + 1):
        for dx in range(-r, r + 1):
            for dy in (-r, r):
                cand = (sx + dx, sy + dy)
                if not valid(cand): continue
                wx, wy = _grid_to_world(cand[0], cand[1])
                if _point_inside_arena(wx, wy, margin=clearance) and not _point_blocked_xy(wx, wy, clearance):
                    return cand
        for dy in range(-r + 1, r):
            for dx in (-r, r):
                cand = (sx + dx, sy + dy)
                if not valid(cand): continue
                wx, wy = _grid_to_world(cand[0], cand[1])
                if _point_inside_arena(wx, wy, margin=clearance) and not _point_blocked_xy(wx, wy, clearance):
                    return cand
    return None

def astar_path(start_xy, goal_xy, clearance):
    nx, ny = _grid_dims()
    start = _nearest_free_grid(_world_to_grid(start_xy[0], start_xy[1]), clearance)
    goal  = _nearest_free_grid(_world_to_grid(goal_xy[0], goal_xy[1]), clearance)
    if start is None or goal is None: return []

    blocked_cache = {}
    def in_bounds(node): return 0 <= node[0] < nx and 0 <= node[1] < ny
    def blocked(node):
        if node in blocked_cache: return blocked_cache[node]
        wx, wy = _grid_to_world(node[0], node[1])
        bad = (not _point_inside_arena(wx, wy, margin=clearance)) or _point_blocked_xy(wx, wy, clearance)
        blocked_cache[node] = bad
        return bad
    def heuristic(a, b): return math.hypot(a[0] - b[0], a[1] - b[1])

    moves = [(-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
             (-1, -1, PLANNER_DIAG_COST), (-1, 1, PLANNER_DIAG_COST),
             (1, -1, PLANNER_DIAG_COST), (1, 1, PLANNER_DIAG_COST)]

    open_heap = []
    heapq.heappush(open_heap, (heuristic(start, goal), 0.0, start))
    parent, g_score = {}, {start: 0.0}
    closed = set()

    while open_heap:
        _, g_cur, cur = heapq.heappop(open_heap)
        if cur in closed: continue
        if cur == goal: break
        closed.add(cur)

        for dx, dy, step_cost in moves:
            nxt = (cur[0] + dx, cur[1] + dy)
            if not in_bounds(nxt) or blocked(nxt) or nxt in closed: continue
            if dx != 0 and dy != 0:
                if blocked((cur[0] + dx, cur[1])) or blocked((cur[0], cur[1] + dy)): continue

            tentative_g = g_cur + step_cost
            if tentative_g < g_score.get(nxt, 1e18):
                g_score[nxt] = tentative_g
                parent[nxt] = cur
                f = tentative_g + heuristic(nxt, goal)
                heapq.heappush(open_heap, (f, tentative_g, nxt))

    if goal not in parent and goal != start: return []
    path_grid = [goal]
    while path_grid[-1] != start: path_grid.append(parent[path_grid[-1]])
    path_grid.reverse()
    return [_grid_to_world(gx, gy) for gx, gy in path_grid]

def simplify_path(path, clearance):
    if len(path) <= 2: return path
    out = [path[0]]
    i = 0
    while i < len(path) - 1:
        j = len(path) - 1
        while j > i + 1 and not _line_clear(path[i], path[j], clearance): j -= 1
        out.append(path[j])
        i = j
    return out

def densify_path(path, max_step=1.05):
    if len(path) < 2: return path
    out = [path[0]]
    for i in range(len(path) - 1):
        a, b = path[i], path[i + 1]
        seg = math.hypot(b[0] - a[0], b[1] - a[1])
        if seg < 1e-6: continue
        pieces = max(1, int(math.ceil(seg / max_step)))
        for k in range(1, pieces + 1):
            t = k / pieces
            out.append((a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t))
    return out

def plan_route(start_xy, goal_xy):
    clearance = ROBOT_RADIUS + PLANNER_MARGIN
    raw_path = astar_path(start_xy, goal_xy, clearance)
    if not raw_path: return [goal_xy]
    simplified = simplify_path(raw_path, clearance)
    dense = densify_path(simplified, max_step=1.05)
    if dense and math.hypot(dense[0][0] - start_xy[0], dense[0][1] - start_xy[1]) < 0.45:
        dense = dense[1:]
    if not dense or math.hypot(dense[-1][0] - goal_xy[0], dense[-1][1] - goal_xy[1]) > 0.25:
        dense.append(goal_xy)
    return dense

def choose_sample_dock(sample_xy, reference_xy):
    best, best_score = None, -1e18
    for i in range(DOCK_CANDIDATE_COUNT):
        ang = (2.0 * math.pi * i) / DOCK_CANDIDATE_COUNT
        dock = (sample_xy[0] + DOCK_RING_RADIUS * math.cos(ang),
                sample_xy[1] + DOCK_RING_RADIUS * math.sin(ang))
        d_sample = math.hypot(dock[0] - sample_xy[0], dock[1] - sample_xy[1])
        
        if d_sample < DOCK_MIN_SAMPLE_DIST or d_sample > DOCK_MAX_SAMPLE_DIST: continue
        if not _point_inside_arena(dock[0], dock[1], margin=ROBOT_RADIUS): continue
        if _point_blocked_xy(dock[0], dock[1], ROBOT_RADIUS + 0.08): continue

        path_probe = astar_path(reference_xy, dock, ROBOT_RADIUS + PLANNER_MARGIN)
        if not path_probe: continue

        to_sample = math.atan2(sample_xy[1] - dock[1], sample_xy[0] - dock[0])
        to_goal = math.atan2(GOAL[1] - dock[1], GOAL[0] - dock[0])
        heading_score = math.cos(_wrap_angle(to_sample - to_goal))
        score = -len(path_probe) + 0.25 * heading_score
        
        if score > best_score:
            best_score = score
            best = dock
    return best if best else (sample_xy[0] - 0.90, sample_xy[1] - 0.75)

# ----------------------------
# Scene setup
# ----------------------------
def load_scene():
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.loadURDF("plane.urdf")

    for pos in OBSTACLE_CUBES:
        col = p.createCollisionShape(p.GEOM_BOX, halfExtents=list(CUBE_HALF_EXTENTS))
        vis = p.createVisualShape(p.GEOM_BOX, halfExtents=list(CUBE_HALF_EXTENTS), rgbaColor=[0.8, 0.3, 0.3, 1])
        p.createMultiBody(0, col, vis, basePosition=pos)

    for pos in CYLINDER_OBSTACLES:
        col = p.createCollisionShape(p.GEOM_CYLINDER, radius=pos[2], height=CYLINDER_HEIGHT)
        vis = p.createVisualShape(p.GEOM_CYLINDER, radius=pos[2], length=CYLINDER_HEIGHT, rgbaColor=[0.4, 0.4, 0.8, 1])
        p.createMultiBody(0, col, vis, basePosition=[pos[0], pos[1], CYLINDER_HEIGHT * 0.5])

    pad_col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.6, 0.6, 0.01])
    pad_vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.6, 0.6, 0.01], rgbaColor=[0.2, 0.9, 0.2, 0.8])
    p.createMultiBody(0, pad_col, pad_vis, basePosition=[GOAL[0], GOAL[1], 0.01])

    husky  = p.loadURDF("husky/husky.urdf", START, useFixedBase=False)
    wheels = [2, 3, 4, 5]
    p.changeDynamics(husky, -1, linearDamping=0.02, angularDamping=0.02, mass=38.0)
    for w in wheels: p.changeDynamics(husky, w, lateralFriction=1.25, spinningFriction=0.01, rollingFriction=0.004)

    kuka = p.loadURDF("kuka_iiwa/model.urdf", basePosition=[START[0], START[1], START[2] + ARM_MOUNT_Z], useFixedBase=False)
    p.createConstraint(husky, -1, kuka, -1, p.JOINT_FIXED, [0, 0, 0], [0, 0, ARM_MOUNT_Z], [0, 0, 0])
    ee_link = p.getNumJoints(kuka) - 1
    for j in range(-1, p.getNumJoints(kuka)): p.changeDynamics(kuka, j, linearDamping=0.15, angularDamping=0.15, mass=1.2)

    block_col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[SAMPLE_HALF_EXTENT, SAMPLE_HALF_EXTENT, SAMPLE_HALF_EXTENT])
    block_vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[SAMPLE_HALF_EXTENT, SAMPLE_HALF_EXTENT, SAMPLE_HALF_EXTENT], rgbaColor=[0.95, 0.8, 0.1, 1])
    sample_id = p.createMultiBody(0.16, block_col, block_vis, basePosition=SAMPLE_POS)

    return husky, wheels, kuka, ee_link, sample_id

# ----------------------------
# Navigation helpers
# ----------------------------
def align_base_yaw(husky, wheels, target_yaw, max_steps=180, tol=0.05):
    for _ in range(max_steps):
        _, orn = p.getBasePositionAndOrientation(husky)
        yaw    = p.getEulerFromQuaternion(orn)[2]
        err    = _wrap_angle(target_yaw - yaw)
        if abs(err) < tol: break
        turn = max(-4.0, min(4.0, err * 6.0))
        p.setJointMotorControlArray(husky, wheels, p.VELOCITY_CONTROL, targetVelocities=[-turn, turn, -turn, turn], forces=[WHEEL_FORCE] * 4)
        sim_step()
    brake(husky, wheels, force_scale=2.0)

def refresh_waypoint_queue(queue, pos_xy):
    if not queue: return queue
    while len(queue) > 1 and math.hypot(queue[0][0] - pos_xy[0], queue[0][1] - pos_xy[1]) < PATH_REACHED_DIST:
        queue.pop(0)
    clearance = ROBOT_RADIUS + 0.20
    while len(queue) > 1 and _line_clear(pos_xy, queue[1], clearance):
        queue.pop(0)
    return queue

def drive_toward(husky, wheels, target):
    """Pure Pursuit driving: Strictly follows A* point without reactive panicking"""
    if not hasattr(drive_toward, "last"): drive_toward.last = [0.0, 0.0, 0.0, 0.0]

    pos, orn = p.getBasePositionAndOrientation(husky)
    yaw      = p.getEulerFromQuaternion(orn)[2]
    dx, dy   = target[0] - pos[0], target[1] - pos[1]
    dist     = math.hypot(dx, dy)
    goal_yaw = math.atan2(dy, dx)

    speed = BASE_SPEED * max(0.40, min(1.0, dist / 2.0))
    err = _wrap_angle(goal_yaw - yaw)
    
    # Slow down if sharp turn is needed to stay on path
    if abs(err) > 0.5:
        speed *= 0.4
        
    w = TURN_GAIN * err * 1.5
    left  = max(-12, min(12, speed - w))
    right = max(-12, min(12, speed + w))

    alpha = DRIVE_SMOOTH
    smooth_fl = alpha * drive_toward.last[0] + (1 - alpha) * left
    smooth_fr = alpha * drive_toward.last[1] + (1 - alpha) * right
    smooth_rl = alpha * drive_toward.last[2] + (1 - alpha) * left
    smooth_rr = alpha * drive_toward.last[3] + (1 - alpha) * right

    def clamp_delta(target_v, prev, max_delta): return max(prev - max_delta, min(prev + max_delta, target_v))
    new_fl = clamp_delta(smooth_fl, drive_toward.last[0], MAX_WHEEL_DELTA)
    new_fr = clamp_delta(smooth_fr, drive_toward.last[1], MAX_WHEEL_DELTA)
    new_rl = clamp_delta(smooth_rl, drive_toward.last[2], MAX_WHEEL_DELTA)
    new_rr = clamp_delta(smooth_rr, drive_toward.last[3], MAX_WHEEL_DELTA)

    drive_toward.last = [new_fl, new_fr, new_rl, new_rr]
    p.setJointMotorControlArray(husky, wheels, p.VELOCITY_CONTROL, targetVelocities=drive_toward.last, forces=[WHEEL_FORCE] * 4)
    return dist

def brake(husky, wheels, force_scale=4.0):
    p.setJointMotorControlArray(husky, wheels, p.VELOCITY_CONTROL, targetVelocities=[0, 0, 0, 0], forces=[WHEEL_FORCE * force_scale] * 4)
    drive_toward.last = [0.0, 0.0, 0.0, 0.0]

def add_base_damping(husky):
    p.changeDynamics(husky, -1, linearDamping=0.06, angularDamping=0.08)
    for w in [2, 3, 4, 5]: p.changeDynamics(husky, w, lateralFriction=1.35, spinningFriction=0.02, rollingFriction=0.01)

# ----------------------------
# Arm helpers
# ----------------------------
def ik_to(pt, kuka, ee_link):
    orn = p.getQuaternionFromEuler([-1.35, 0, 0])
    return p.calculateInverseKinematics(kuka, ee_link, pt, orn, maxNumIterations=180, residualThreshold=1e-4, jointDamping=ARM_DAMPING, restPoses=ARM_REST)

def set_arm(kuka, joints):
    n = min(p.getNumJoints(kuka), len(joints))
    for j in range(n):
        p.setJointMotorControl2(kuka, j, p.POSITION_CONTROL, targetPosition=joints[j], positionGain=ARM_POS_GAIN, velocityGain=ARM_VEL_GAIN, maxVelocity=ARM_MAX_VEL, force=ARM_FORCE)

def smooth_arm_to_joints(kuka, target_joints, steps=80):
    start = [p.getJointState(kuka, j)[0] for j in range(p.getNumJoints(kuka))]
    n     = min(len(start), len(target_joints))
    for k in range(steps):
        t        = (k + 1) / steps
        smooth_t = 3 * t * t - 2 * t * t * t
        joints   = [(1 - smooth_t) * start[j] + smooth_t * target_joints[j] for j in range(n)]
        set_arm(kuka, joints)
        sim_step()

def get_ee_pos(kuka, ee_link):
    state = p.getLinkState(kuka, ee_link)
    return state[0]

def smooth_arm_move(kuka, ee_link, pts, steps=120):
    """Restored Cartesian interpolation to prevent joint-flipping errors."""
    curr_pos = get_ee_pos(kuka, ee_link)
    full_pts = [curr_pos] + pts

    for i in range(len(full_pts) - 1):
        start_pt = full_pts[i]
        end_pt = full_pts[i + 1]

        for k in range(steps):
            t = k / float(steps)
            smooth_t = 3 * (t ** 2) - 2 * (t ** 3)
            curr_pt = [start_pt[j] + smooth_t * (end_pt[j] - start_pt[j]) for j in range(3)]
            joints = ik_to(curr_pt, kuka, ee_link)
            set_arm(kuka, joints)
            sim_step()
        sim_step(18)

# ----------------------------
# Main
# ----------------------------
def main():
    global SIM_REALTIME
    gui_requested = os.environ.get("PYBULLET_GUI", "1").lower() not in {"0", "false", "no"}
    connect_mode = p.GUI if gui_requested else p.DIRECT
    cid = p.connect(connect_mode)
    if cid < 0 and connect_mode == p.GUI:
        cid = p.connect(p.DIRECT)
        connect_mode = p.DIRECT
    SIM_REALTIME = (connect_mode == p.GUI) and (os.environ.get("PYBULLET_REALTIME", "1").lower() not in {"0", "false", "no"})

    p.setGravity(0, 0, -9.8)
    p.setTimeStep(TIME_STEP)
    if connect_mode == p.GUI: p.resetDebugVisualizerCamera(14, 45, -35, [0, 0, 0])

    husky, wheels, kuka, ee_link, sample_id = load_scene()
    add_base_damping(husky)
    sim_step(240)
    set_arm(kuka, ARM_STOW_JOINTS)

    phase = Phase.NAV_TO_SAMPLE
    start_xy = (START[0], START[1])
    goal_xy = (GOAL[0], GOAL[1])
    sample_pos, _ = p.getBasePositionAndOrientation(sample_id)
    sample_xy = (sample_pos[0], sample_pos[1])
    
    sample_dock = choose_sample_dock(sample_xy, start_xy)
    nav_queue = plan_route(start_xy, sample_dock)
    if not nav_queue: nav_queue = [sample_dock]
    
    goal_queue = []
    constraint_id = None
    t0 = time.time()

    while True:
        if time.time() - t0 > 200:
            print("Timeout; stopping.")
            break

        # ── Navigate to sample ─────────────────────────────────────────────
        if phase == Phase.NAV_TO_SAMPLE:
            set_arm(kuka, ARM_STOW_JOINTS)
            pos, _ = p.getBasePositionAndOrientation(husky)
            nav_queue = refresh_waypoint_queue(nav_queue, (pos[0], pos[1]))
            
            if not nav_queue:
                brake(husky, wheels)
                sim_step(120)
                sample_pos, _ = p.getBasePositionAndOrientation(sample_id)
                yaw_to_sample = math.atan2(sample_pos[1] - pos[1], sample_pos[0] - pos[0])
                align_base_yaw(husky, wheels, yaw_to_sample, tol=0.035)
                phase = Phase.PICK
                continue

            target_xy = nav_queue[0]
            dist      = drive_toward(husky, wheels, target_xy)
            sim_step()

            reach_thresh = PATH_REACHED_DIST
            if dist < reach_thresh:
                nav_queue.pop(0)

        # ── Pick ───────────────────────────────────────────────────────────
        elif phase == Phase.PICK:
            sample_pos, _ = p.getBasePositionAndOrientation(sample_id)
            approach = (sample_pos[0], sample_pos[1], sample_pos[2] + 0.22)
            grasp    = (sample_pos[0], sample_pos[1], sample_pos[2] + 0.03)
            smooth_arm_move(kuka, ee_link, [approach, grasp], steps=100)

            constraint_id = p.createConstraint(kuka, ee_link, sample_id, -1, p.JOINT_FIXED, [0, 0, 0], [0, 0, 0.03], [0, 0, 0])
            p.changeConstraint(constraint_id, maxForce=500)
            smooth_arm_to_joints(kuka, ARM_CARRY_JOINTS, steps=85)

            phase = Phase.NAV_TO_GOAL
            base_pos, _ = p.getBasePositionAndOrientation(husky)
            goal_queue = plan_route((base_pos[0], base_pos[1]), goal_xy)
            if not goal_queue: goal_queue = [goal_xy]
            brake(husky, wheels)

        # ── Navigate to goal ───────────────────────────────────────────────
        elif phase == Phase.NAV_TO_GOAL:
            set_arm(kuka, ARM_CARRY_JOINTS)
            pos, _ = p.getBasePositionAndOrientation(husky)
            goal_queue = refresh_waypoint_queue(goal_queue, (pos[0], pos[1]))
            
            if not goal_queue:
                brake(husky, wheels)
                sim_step(120)
                phase = Phase.PLACE
                continue

            target_xy = goal_queue[0]
            dist      = drive_toward(husky, wheels, target_xy)
            sim_step()

            reach_thresh = 0.6 if len(goal_queue) == 1 else PATH_REACHED_DIST
            if dist < reach_thresh:
                goal_queue.pop(0)

        # ── Place & Exit cleanly────────────────────────────────────────────
        elif phase == Phase.PLACE:
            # Place block dynamically relative to wherever the robot successfully stopped
            pos, orn = p.getBasePositionAndOrientation(husky)
            yaw = p.getEulerFromQuaternion(orn)[2]
            
            drop_x = pos[0] + math.cos(yaw) * 0.65
            drop_y = pos[1] + math.sin(yaw) * 0.65
            
            drop_over = (drop_x, drop_y, 0.30)
            drop      = (drop_x, drop_y, 0.10)
            
            smooth_arm_move(kuka, ee_link, [drop_over, drop], steps=100)
            if constraint_id is not None:
                p.removeConstraint(constraint_id)
            
            smooth_arm_to_joints(kuka, ARM_STOW_JOINTS, steps=70)
            phase = Phase.DONE

        # ── Done ───────────────────────────────────────────────────────────
        elif phase == Phase.DONE:
            # Immediate exit when the green tile delivery is successfully met
            break

    p.disconnect()

if __name__ == "__main__":
    main()
