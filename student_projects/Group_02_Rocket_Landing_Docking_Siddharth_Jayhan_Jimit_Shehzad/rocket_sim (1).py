import pybullet as p
import pybullet_data
import time
import math

# ---------------- SETUP ----------------
p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.81)

p.loadURDF("plane.urdf")

# ---------------- SLIDERS ----------------
kp_slider = p.addUserDebugParameter("Kp (Tilt)", 0, 10, 4)
kd_slider = p.addUserDebugParameter("Kd (Damping)", 0, 5, 1.5)
wind_slider = p.addUserDebugParameter("Wind", 0, 5, 0.5)

# ---------------- LAUNCHPAD ----------------
p.createMultiBody(
    0,
    p.createCollisionShape(p.GEOM_BOX, halfExtents=[1.5,1.5,0.1]),
    p.createVisualShape(p.GEOM_BOX, halfExtents=[1.5,1.5,0.1], rgbaColor=[0.2,0.2,0.2,1]),
    [0,0,0.1]
)

# ---------------- CLAWS ----------------
claw_open, claw_closed = 0.8, 0.35
claw_pos = claw_open

claw_left = p.createMultiBody(
    0,
    p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.05,0.3,0.5]),
    p.createVisualShape(p.GEOM_BOX, halfExtents=[0.05,0.3,0.5], rgbaColor=[0.7,0.7,0.7,1]),
    [-claw_pos,0,0.6]
)

claw_right = p.createMultiBody(
    0,
    p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.05,0.3,0.5]),
    p.createVisualShape(p.GEOM_BOX, halfExtents=[0.05,0.3,0.5], rgbaColor=[0.7,0.7,0.7,1]),
    [claw_pos,0,0.6]
)

# ---------------- ROCKET WITH LEGS ----------------
visual_shapes = p.createVisualShapeArray(
    shapeTypes=[p.GEOM_CYLINDER]*4 + [p.GEOM_BOX]*8,
    radii=[0.07,0.05,0.03,0.015] + [0]*8,
    lengths=[0.6,0.12,0.10,0.08] + [0]*8,
    halfExtents=[[0,0,0]]*4 + [
        [0.02,0.12,0.08],[0.02,0.12,0.08],[0.12,0.02,0.08],[0.12,0.02,0.08],
        [0.03,0.03,0.35],[0.03,0.03,0.35],[0.03,0.03,0.35],[0.03,0.03,0.35]
    ],
    visualFramePositions=[
        [0,0,0],[0,0,0.40],[0,0,0.50],[0,0,0.60],
        [0.08,0,-0.2],[-0.08,0,-0.2],[0,0.08,-0.2],[0,-0.08,-0.2],
        [0.25,0.25,-0.5],[-0.25,0.25,-0.5],[0.25,-0.25,-0.5],[-0.25,-0.25,-0.5]
    ],
    rgbaColors=[[1,1,1,1]] + [[1,0,0,1]]*3 + [[0.8,0,0,1]]*4 + [[0.3,0.3,0.3,1]]*4
)

rocket = p.createMultiBody(
    baseMass=1,
    baseCollisionShapeIndex=p.createCollisionShape(p.GEOM_CYLINDER, radius=0.07, height=0.6),
    baseVisualShapeIndex=visual_shapes,
    basePosition=[0,0,1]
)

p.changeDynamics(rocket, -1, linearDamping=0.2, angularDamping=1.5)

# ---------------- VARIABLES ----------------
state = "lift"
target_height = 50
hover_start = None
start_time = time.time()

target_pos = [0,0]

M_TO_FT = 3.28084
ground_offset = 0.6

hud_ids = []
flame_ids = []

# ---------------- LOOP ----------------
while True:

    t = time.time() - start_time

    kp = p.readUserDebugParameter(kp_slider)
    kd = p.readUserDebugParameter(kd_slider)
    wind_strength = p.readUserDebugParameter(wind_slider)

    pos, orn = p.getBasePositionAndOrientation(rocket)
    vel, ang_vel = p.getBaseVelocity(rocket)

    height = max(0, pos[2] - ground_offset)

    vx, vy, vz = vel
    speed = math.sqrt(vx*vx + vy*vy + vz*vz)

    roll, pitch, yaw = p.getEulerFromQuaternion(orn)

    # -------- PID --------
    gimbal_x = kp*(-pitch) - kd*ang_vel[1]
    gimbal_y = kp*(-roll) - kd*ang_vel[0]

    # -------- STATES --------
    if state == "lift":
        thrust = 35
        if height > target_height:
            state = "hover"
            hover_start = time.time()

    elif state == "hover":
        thrust = 18*(target_height-height) - 6*vz + 9.81
        if time.time() - hover_start > 2:
            state = "dock"

    elif state == "dock":
        err_x, err_y = -pos[0], -pos[1]

        gimbal_x += max(min(0.3*err_y,0.2),-0.2)
        gimbal_y += max(min(-0.3*err_x,0.2),-0.2)

        thrust = 10*(-height) - 4*vz + 9.81

        if height < 20:
            thrust += 6

        if height < 1:
            thrust = 9.81*0.95

        if height < 0.05 and abs(vz) < 0.2:
            state = "landed"

    else:
        thrust = 0

    thrust = max(0, min(thrust, 50))

    # -------- THRUST VECTOR --------
    thrust_dir = [gimbal_x, -gimbal_y, 1]
    mag = math.sqrt(sum(i*i for i in thrust_dir))
    thrust_dir = [i/mag for i in thrust_dir]

    # -------- FORCES --------
    drag = [-0.2*vx, -0.2*vy, -0.2*vz]

    wind = [
        wind_strength*(math.sin(t)+0.5*math.sin(3*t)),
        wind_strength*(math.cos(t*0.5)+0.3*math.sin(2*t)),
        0
    ]

    thrust_force = [thrust*d for d in thrust_dir]

    total_force = [
        thrust_force[0]+drag[0]+wind[0],
        thrust_force[1]+drag[1]+wind[1],
        thrust_force[2]+drag[2]
    ]

    p.applyExternalForce(rocket, -1, total_force, pos, p.WORLD_FRAME)

    # -------- CINEMATIC CAMERA --------
    if state == "lift":
        p.resetDebugVisualizerCamera(
            6 + 2*math.sin(t*0.5),
            t*40,
            -30 + 10*math.sin(t*0.3),
            pos
        )

    elif state == "hover":
        p.resetDebugVisualizerCamera(10, 45, -20, pos)

    elif state == "dock":
        if height > 10:
            p.resetDebugVisualizerCamera(
                8 + 1.5*math.sin(t*0.4),
                60 + 10*math.sin(t*0.2),
                -25,
                pos
            )
        else:
            p.resetDebugVisualizerCamera(
                4,
                30,
                -10,
                [pos[0], pos[1], pos[2]*0.5]
            )

    elif state == "landed":
        p.resetDebugVisualizerCamera(5, 30, -10, [0,0,1])

    # -------- CLAWS --------
    err = math.sqrt(pos[0]**2 + pos[1]**2)

    target_claw = claw_closed if (state=="dock" and height<3 and err<0.5) else claw_open
    claw_pos += (target_claw - claw_pos)*0.1

    p.resetBasePositionAndOrientation(claw_left, [-claw_pos,0,0.6],[0,0,0,1])
    p.resetBasePositionAndOrientation(claw_right, [claw_pos,0,0.6],[0,0,0,1])
    
        # -------- CONTACT (RAIL BEHAVIOR) --------
    contacts = p.getContactPoints(rocket, claw_left) + p.getContactPoints(rocket, claw_right)

    # Touching claw → remove sideways motion, allow falling
    if len(contacts) > 0 and state == "dock":
        vx, vy, vz = vel
        p.resetBaseVelocity(rocket, [0, 0, vz], [0,0,0])

    # Only lock near ground
    if len(contacts) > 0 and state == "dock" and height < 0.2:
        p.resetBaseVelocity(rocket, [0,0,0], [0,0,0])
        p.changeDynamics(rocket, -1, linearDamping=0.9, angularDamping=0.9)



    # -------- FLAME --------
    for fid in flame_ids:
        p.removeUserDebugItem(fid)
    flame_ids = []

    if thrust > 0:
        flame_len = (thrust/50) + abs(vz)*0.1
        flicker = 0.8 + 0.4*math.sin(20*t)
        flame_len *= flicker

        for i in range(3):
            start = [pos[0],pos[1],pos[2]-0.2]
            end = [
                start[0]-thrust_dir[0]*flame_len+(i-1)*0.02,
                start[1]-thrust_dir[1]*flame_len,
                start[2]-thrust_dir[2]*flame_len
            ]
            flame_ids.append(p.addUserDebugLine(start,end,[1,0.5,0],3))

    # -------- HUD --------
    for hid in hud_ids:
        p.removeUserDebugItem(hid)
    hud_ids = []

    hud_ids.append(p.addUserDebugText(f"Altitude: {height*M_TO_FT:.1f} ft", [-3,2,2],[0,1,0],1.5))
    hud_ids.append(p.addUserDebugText(f"Speed: {speed*M_TO_FT:.1f} ft/s", [-3,2,1.7],[1,1,0],1.5))
    hud_ids.append(p.addUserDebugText(f"Kp:{kp:.2f} Kd:{kd:.2f}", [-3,2,1.4],[1,1,1],1.5))

    if state == "landed":
        p.resetBaseVelocity(rocket,[0,0,0],[0,0,0])

    p.stepSimulation()
    time.sleep(1/120)