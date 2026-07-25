import sys
import subprocess
import importlib.util
import time
import site

print("=======================================")
print(" Stewart Platform Ball Balancing Sim  ")
print("=======================================")

# ------------------------------------------------
# Ensure user site-packages are accessible
# ------------------------------------------------
user_site = site.getusersitepackages()
if user_site not in sys.path:
    sys.path.append(user_site)
print("\nUser site-packages:", user_site)

# ------------------------------------------------
# Python Version Check
# ------------------------------------------------
print("\nChecking Python version...")
print("Python Version:", sys.version)
if sys.version_info.major == 3 and sys.version_info.minor == 10:
    print("Recommended Python version detected.")
else:
    print("Warning: Python 3.10 recommended.")

# ------------------------------------------------
# Package Installer
# ------------------------------------------------
def install(package):
    print(f"\nInstalling {package} ...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--user", package])

# ------------------------------------------------
# Check & Install Required Packages
# ------------------------------------------------
required_packages = ["pybullet", "numpy"]
installed_any = False
for pkg in required_packages:
    if importlib.util.find_spec(pkg) is None:
        install(pkg)
        installed_any = True
    else:
        print(f"{pkg} already installed.")

# Refresh sys.path so newly installed user packages are importable
if installed_any:
    import importlib
    importlib.invalidate_caches()
    import site as _site
    _site.main()

print("\nAll package checks completed.")

# ------------------------------------------------
# Import Libraries
# ------------------------------------------------
print("\nImporting libraries...")
import pybullet as p
import pybullet_data
import numpy as np
print("All libraries imported successfully.")

# ------------------------------------------------
# Launch Simulation
# ------------------------------------------------
print("\nLaunching PyBullet GUI...")
physicsClient = p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.81)
DT = 1.0 / 240.0
p.setTimeStep(DT)

p.resetDebugVisualizerCamera(
    cameraDistance=1.1,
    cameraYaw=45,
    cameraPitch=-25,
    cameraTargetPosition=[0, 0, 0.2])
p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 1)
p.configureDebugVisualizer(p.COV_ENABLE_GUI, 1)
p.configureDebugVisualizer(p.COV_ENABLE_MOUSE_PICKING, 1)

# ------------------------------------------------
# Load Environment
# ------------------------------------------------
plane = p.loadURDF("plane.urdf")
print("Environment loaded successfully.")

# ------------------------------------------------
# Stewart Platform Geometry
# ------------------------------------------------
N      = 3
R_BASE = 0.22
R_PLAT = 0.14
GAMMA  = 0.0
H0     = 0.28
L_HORN = 0.04
L_ROD  = 0.23

servo_pos = [np.array([R_BASE*np.cos(2*np.pi*i/N),
                        R_BASE*np.sin(2*np.pi*i/N), 0.0]) for i in range(N)]

plat_joints_local = [np.array([R_PLAT*np.cos(2*np.pi*i/N + GAMMA),
                                 R_PLAT*np.sin(2*np.pi*i/N + GAMMA), 0.0]) for i in range(N)]

# ------------------------------------------------
# IK & Math Helpers
# ------------------------------------------------
def solve_servo_angle(i, plat_pos, R_mat):
    phi   = 2*np.pi*i/N
    r_hat = np.array([np.cos(phi), np.sin(phi), 0.0])
    P     = servo_pos[i]
    Q     = np.array(plat_pos) + R_mat @ plat_joints_local[i]
    PQ    = Q - P
    A     = np.dot(PQ, r_hat)
    B     = PQ[2]
    C     = (L_HORN**2 + np.dot(PQ,PQ) - L_ROD**2) / (2*L_HORN)
    denom = np.sqrt(A**2 + B**2)
    if denom < 1e-9: return 0.0
    ratio = np.clip(C / denom, -1.0, 1.0)
    return float(np.arctan2(B, A) - np.arccos(ratio))

def horn_tip_world(i, theta):
    phi   = 2*np.pi*i/N
    r_hat = np.array([np.cos(phi), np.sin(phi), 0.0])
    return servo_pos[i] + L_HORN*(np.cos(theta)*r_hat + np.array([0,0,np.sin(theta)]))

def tilt_to_rot(alpha, beta):
    ca, sa = np.cos(alpha), np.sin(alpha)
    cb, sb = np.cos(beta),  np.sin(beta)
    return np.array([[ cb,    sa*sb,  ca*sb],
                     [ 0,     ca,    -sa   ],
                     [-sb,    sa*cb,  ca*cb]])

def rot_to_quat(R):
    tr = R[0,0]+R[1,1]+R[2,2]
    if tr > 0:
        s=0.5/np.sqrt(tr+1); w=0.25/s
        x=(R[2,1]-R[1,2])*s; y=(R[0,2]-R[2,0])*s; z=(R[1,0]-R[0,1])*s
    elif R[0,0]>R[1,1] and R[0,0]>R[2,2]:
        s=2*np.sqrt(1+R[0,0]-R[1,1]-R[2,2])
        w=(R[2,1]-R[1,2])/s; x=0.25*s; y=(R[0,1]+R[1,0])/s; z=(R[0,2]+R[2,0])/s
    elif R[1,1]>R[2,2]:
        s=2*np.sqrt(1+R[1,1]-R[0,0]-R[2,2])
        w=(R[0,2]-R[2,0])/s; x=(R[0,1]+R[1,0])/s; y=0.25*s; z=(R[1,2]+R[2,1])/s
    else:
        s=2*np.sqrt(1+R[2,2]-R[0,0]-R[1,1])
        w=(R[1,0]-R[0,1])/s; x=(R[0,2]+R[2,0])/s; y=(R[1,2]+R[2,1])/s; z=0.25*s
    return [x,y,z,w]

def cylinder_pose(pt_a, pt_b):
    mid = (np.array(pt_a) + np.array(pt_b)) / 2
    vec = np.array(pt_b) - np.array(pt_a)
    L   = np.linalg.norm(vec)
    if L < 1e-9: return mid.tolist(), [0,0,0,1]
    d     = vec / L
    cross = np.cross([0,0,1], d)
    cl    = np.linalg.norm(cross)
    if cl < 1e-9:
        q = [0,0,0,1] if d[2]>0 else [1,0,0,0]
    else:
        ax=cross/cl; ang=np.arccos(np.clip(np.dot([0,0,1],d),-1,1)); h=ang/2
        q=[ax[0]*np.sin(h), ax[1]*np.sin(h), ax[2]*np.sin(h), np.cos(h)]
    return mid.tolist(), q

# ------------------------------------------------
# Build Scene
# ------------------------------------------------

# Base plate
p.createMultiBody(
    baseMass=0,
    baseCollisionShapeIndex=p.createCollisionShape(p.GEOM_CYLINDER, radius=0.28, height=0.015),
    baseVisualShapeIndex   =p.createVisualShape   (p.GEOM_CYLINDER, radius=0.28, length=0.015,
                                                   rgbaColor=[0.15,0.15,0.15,1]),
    basePosition=[0, 0, 0.0075])

# Servo bodies
for i in range(N):
    phi = 2*np.pi*i/N
    p.createMultiBody(
        baseMass=0,
        baseVisualShapeIndex=p.createVisualShape(p.GEOM_BOX,
                                                  halfExtents=[0.018,0.012,0.016],
                                                  rgbaColor=[0.1,0.1,0.1,1]),
        basePosition=servo_pos[i].tolist(),
        baseOrientation=p.getQuaternionFromEuler([0,0,phi]))

# Platform (semi-transparent)
platform = p.createMultiBody(
    baseMass=0,
    baseCollisionShapeIndex=p.createCollisionShape(p.GEOM_CYLINDER, radius=0.17, height=0.008),
    baseVisualShapeIndex   =p.createVisualShape   (p.GEOM_CYLINDER, radius=0.17, length=0.008,
                                                   rgbaColor=[0.9, 0.2, 0.2, 0.45]),
    basePosition=[0, 0, H0])

# Horn visuals (green)
horn_bodies = []
for i in range(N):
    hb = p.createMultiBody(
        baseMass=0,
        baseVisualShapeIndex=p.createVisualShape(p.GEOM_CYLINDER, radius=0.005,
                                                  length=L_HORN,
                                                  rgbaColor=[0.0, 0.85, 0.2, 1.0]),
        basePosition=servo_pos[i].tolist())
    horn_bodies.append(hb)

# Pushrod visuals (yellow)
rod_bodies = []
for i in range(N):
    rb = p.createMultiBody(
        baseMass=0,
        baseVisualShapeIndex=p.createVisualShape(p.GEOM_CYLINDER, radius=0.004,
                                                  length=L_ROD,
                                                  rgbaColor=[0.9, 0.75, 0.1, 1.0]),
        basePosition=servo_pos[i].tolist())
    rod_bodies.append(rb)

# Target marker — cyan sphere at centre
target_marker = p.createMultiBody(
    baseMass=0,
    baseVisualShapeIndex=p.createVisualShape(p.GEOM_SPHERE, radius=0.014,
                                              rgbaColor=[0.0, 1.0, 1.0, 1.0]),
    basePosition=[0, 0, H0 + 0.012])

# Ball
BALL_R = 0.020
ball = p.createMultiBody(
    baseMass=0.0027,
    baseCollisionShapeIndex=p.createCollisionShape(p.GEOM_SPHERE, radius=BALL_R),
    baseVisualShapeIndex   =p.createVisualShape   (p.GEOM_SPHERE, radius=BALL_R,
                                                   rgbaColor=[1, 1, 1, 1]),
    basePosition=[0.0, 0.0, H0 + BALL_R + 0.30])

# Randomise starting X,Y within platform radius
import random
_angle = random.uniform(0, 2 * np.pi)
_radius = random.uniform(0, 0.10)          # stay within platform (R_PLAT ≈ 0.14)
_start_x = _radius * np.cos(_angle)
_start_y = _radius * np.sin(_angle)
p.resetBasePositionAndOrientation(
    ball, [_start_x, _start_y, H0 + BALL_R + 0.30], [0, 0, 0, 1])
p.resetBaseVelocity(ball, linearVelocity=[0, 0, 0])
p.changeDynamics(ball,     -1, lateralFriction=0.5,  restitution=0.6,
                 rollingFriction=0.002, spinningFriction=0.001)
p.changeDynamics(platform, -1, lateralFriction=0.5,  restitution=0.05)

print("Stewart Platform + Ball created successfully.")

# ------------------------------------------------
# Sliders & Controls
# ------------------------------------------------
print("\n" + "="*56)
print("  CONTROLS:")
print("  Arrow Keys   — move the target on the platform")
print("  R key        — reset ball to centre")
print("  PID sliders  — tune controller (right panel)")
print("  Mouse        — orbit/zoom the camera freely")
print("="*56 + "\n")

sid_kp     = p.addUserDebugParameter("Kp",              0.0,  20.0,   8.0)
sid_ki     = p.addUserDebugParameter("Ki",              0.0,   2.0,   0.10)
sid_kd     = p.addUserDebugParameter("Kd",              0.0,  10.0,   2.5)
sid_tilt   = p.addUserDebugParameter("Max Tilt (deg)",  1.0,  25.0,  14.0)

# Reset button (click the slider to trigger — value changes on each click)
sid_reset  = p.addUserDebugParameter(">> RESET BALL <<", 1, 0, 0)
prev_reset_val = p.readUserDebugParameter(sid_reset)

# ------------------------------------------------
# Arrow-key target position
# ------------------------------------------------
# Arrow-key force on ball
# ------------------------------------------------
BALL_PUSH_FORCE = 0.02   # Newtons applied to ball while arrow key is held

# ------------------------------------------------
# Visual helpers
# ------------------------------------------------
def update_visuals(plat_pos, R_mat, thetas):
    for i in range(N):
        tip = horn_tip_world(i, thetas[i])
        top = np.array(plat_pos) + R_mat @ plat_joints_local[i]
        mid, q = cylinder_pose(servo_pos[i], tip)
        p.resetBasePositionAndOrientation(horn_bodies[i], mid, q)
        mid, q = cylinder_pose(tip, top)
        p.resetBasePositionAndOrientation(rod_bodies[i], mid, q)

def relaunch_ball(speed, angle_deg):
    """Reset ball to centre of platform and launch at given speed/angle."""
    angle_rad = np.deg2rad(angle_deg)
    vx = speed * np.cos(angle_rad)
    vy = speed * np.sin(angle_rad)
    p.resetBasePositionAndOrientation(
        ball, [0.0, 0.0, H0 + BALL_R + 0.008], [0,0,0,1])
    p.resetBaseVelocity(ball, linearVelocity=[vx, vy, 0])
    print(f"  >> Ball relaunched at {speed:.2f} m/s, angle {angle_deg:.0f}deg  "
          f"(vx={vx:+.3f}, vy={vy:+.3f})")

def reset_ball():
    """Reset ball to rest directly above platform centre."""
    drop_height = H0 + BALL_R + 0.06   # ~6 cm above platform surface
    p.resetBasePositionAndOrientation(
        ball, [0.0, 0.0, drop_height], [0, 0, 0, 1])
    p.resetBaseVelocity(ball, linearVelocity=[0, 0, 0], angularVelocity=[0, 0, 0])
    print("  >> Ball RESET to centre (no velocity).")

# ------------------------------------------------
# PID state
# ------------------------------------------------
int_ex  = int_ey  = 0.0
prev_ex = prev_ey = 0.0
D_FILTER        = 0.18
filt_dex        = filt_dey = 0.0

# ------------------------------------------------
# Impact detection state
# ------------------------------------------------
# We track the vertical velocity of the ball each step.
# A sudden upward spike (vz going from negative to positive) = bounce/impact.
# On impact we temporarily boost Kp and Kd to recover faster.
prev_vz          = 0.0
impact_boost     = 0.0       # extra gain multiplier (decays each step)
IMPACT_BOOST_MAX = 1.8       # how much extra gain on impact  (1.0 = no boost)
IMPACT_DECAY     = 0.992     # how fast boost fades per step

# ------------------------------------------------
# HUD text IDs
# ------------------------------------------------
hud_pos   = p.addUserDebugText("...", [-0.22, -0.22, H0+0.18], textColorRGB=[1,1,0],   textSize=1.1)
hud_vel   = p.addUserDebugText("...", [-0.22, -0.22, H0+0.14], textColorRGB=[0.4,1,0.4], textSize=1.1)
hud_dist  = p.addUserDebugText("...", [-0.22, -0.22, H0+0.10], textColorRGB=[1,0.6,0.1], textSize=1.1)
hud_tilt  = p.addUserDebugText("...", [-0.22, -0.22, H0+0.06], textColorRGB=[0.6,0.8,1], textSize=1.1)
hud_boost = p.addUserDebugText("...", [-0.22, -0.22, H0+0.02], textColorRGB=[1,0.3,0.3], textSize=1.1)

# ------------------------------------------------
# Animation Loop
# ------------------------------------------------
print("Running simulation...\n")
print("  Ball starts at (0.07, 0.05) with gentle push.")
print("  Use Launch Speed + Launch Angle sliders, then")
print("  press ENTER in this terminal to fire the ball.\n")

# (Terminal relaunch removed — use arrow keys + R key in PyBullet window instead)

step = 0
try:
    while True:

        # ---- check for reset button (slider) ----
        cur_reset_val = p.readUserDebugParameter(sid_reset)
        if cur_reset_val != prev_reset_val:
            prev_reset_val = cur_reset_val
            reset_ball()
            int_ex = int_ey = 0.0
            filt_dex = filt_dey = 0.0

        # ---- arrow keys push the ball / R key resets ----
        keys = p.getKeyboardEvents()
        fx, fy = 0.0, 0.0
        if p.B3G_UP_ARROW in keys and keys[p.B3G_UP_ARROW] & p.KEY_IS_DOWN:
            fy += BALL_PUSH_FORCE
        if p.B3G_DOWN_ARROW in keys and keys[p.B3G_DOWN_ARROW] & p.KEY_IS_DOWN:
            fy -= BALL_PUSH_FORCE
        if p.B3G_RIGHT_ARROW in keys and keys[p.B3G_RIGHT_ARROW] & p.KEY_IS_DOWN:
            fx += BALL_PUSH_FORCE
        if p.B3G_LEFT_ARROW in keys and keys[p.B3G_LEFT_ARROW] & p.KEY_IS_DOWN:
            fx -= BALL_PUSH_FORCE
        if fx != 0.0 or fy != 0.0:
            pos_now, _ = p.getBasePositionAndOrientation(ball)
            p.applyExternalForce(ball, -1, [fx, fy, 0], pos_now, p.WORLD_FRAME)
        # Space = jump in place
        if ord(' ') in keys and keys[ord(' ')] & p.KEY_WAS_TRIGGERED:
            pos_now, _ = p.getBasePositionAndOrientation(ball)
            p.applyExternalForce(ball, -1, [0, 0, 0.6], pos_now, p.WORLD_FRAME)
        # R key = reset ball
        if ord('r') in keys and keys[ord('r')] & p.KEY_WAS_TRIGGERED:
            reset_ball()
            int_ex = int_ey = 0.0
            filt_dex = filt_dey = 0.0

        # ---- read sliders ----
        Kp       = p.readUserDebugParameter(sid_kp)
        Ki       = p.readUserDebugParameter(sid_ki)
        Kd       = p.readUserDebugParameter(sid_kd)
        MAX_TILT = np.deg2rad(p.readUserDebugParameter(sid_tilt))

        # ---- sense ball ----
        pos, _      = p.getBasePositionAndOrientation(ball)
        lin_vel, _  = p.getBaseVelocity(ball)
        bx, by, bz  = float(pos[0]), float(pos[1]), float(pos[2])
        vx, vy, vz  = float(lin_vel[0]), float(lin_vel[1]), float(lin_vel[2])
        speed_xy    = np.sqrt(vx**2 + vy**2)
        speed_total = np.sqrt(vx**2 + vy**2 + vz**2)

        # ---- impact detection ----
        # Ball just bounced if vz flipped from negative to positive
        if prev_vz < -0.05 and vz > 0.0:
            impact_strength = abs(prev_vz)          # m/s downward just before bounce
            impact_boost    = min(IMPACT_BOOST_MAX, 1.0 + impact_strength * 4.0)
            print(f"  [IMPACT] vz_before={prev_vz:.3f} m/s  "
                  f"boost={impact_boost:.2f}x  "
                  f"ball=({bx:+.3f},{by:+.3f})")
        else:
            impact_boost = max(1.0, impact_boost * IMPACT_DECAY)
        prev_vz = vz

        # ---- PID error (always targets centre) ----
        ex = bx
        ey = by

        int_ex = float(np.clip(int_ex + ex * DT, -0.12, 0.12))
        int_ey = float(np.clip(int_ey + ey * DT, -0.12, 0.12))

        raw_dex  = (ex - prev_ex) / DT
        raw_dey  = (ey - prev_ey) / DT
        filt_dex = D_FILTER*filt_dex + (1-D_FILTER)*raw_dex
        filt_dey = D_FILTER*filt_dey + (1-D_FILTER)*raw_dey
        prev_ex, prev_ey = ex, ey

        # ---- PID output with impact boost ----
        # On impact the ball has extra momentum — multiply gains temporarily
        eff_Kp = Kp * impact_boost
        eff_Kd = Kd * impact_boost

        alpha =  (eff_Kp*ey + Ki*int_ey + eff_Kd*filt_dey)
        beta  = -(eff_Kp*ex + Ki*int_ex + eff_Kd*filt_dex)
        alpha  = float(np.clip(alpha, -MAX_TILT, MAX_TILT))
        beta   = float(np.clip(beta,  -MAX_TILT, MAX_TILT))

        # ---- IK → move platform ----
        R        = tilt_to_rot(alpha, beta)
        q        = rot_to_quat(R)
        plat_pos = [0.0, 0.0, H0]
        p.resetBasePositionAndOrientation(platform, plat_pos, q)
        thetas   = [solve_servo_angle(i, plat_pos, R) for i in range(N)]
        update_visuals(plat_pos, R, thetas)

        # ---- step physics ----
        p.stepSimulation()
        time.sleep(1.0 / 240.0)

        # ---- update HUD every 8 steps ----
        if step % 8 == 0:
            dist = np.sqrt(bx**2 + by**2)

            hud_pos = p.addUserDebugText(
                f"Ball  X:{bx:+.4f}m  Y:{by:+.4f}m  Z:{bz:.4f}m",
                [-0.22, -0.22, H0+0.22],
                textColorRGB=[1, 1, 0], textSize=1.1,
                replaceItemUniqueId=hud_pos)

            hud_vel = p.addUserDebugText(
                f"Vel   Vx:{vx:+.3f}  Vy:{vy:+.3f}  |V|:{speed_total:.3f} m/s",
                [-0.22, -0.22, H0+0.18],
                textColorRGB=[0.4, 1, 0.4], textSize=1.1,
                replaceItemUniqueId=hud_vel)

            hud_dist = p.addUserDebugText(
                f"Dist from centre: {dist:.4f}m   XY speed: {speed_xy:.3f} m/s",
                [-0.22, -0.22, H0+0.14],
                textColorRGB=[1, 0.6, 0.1], textSize=1.1,
                replaceItemUniqueId=hud_dist)

            hud_tilt = p.addUserDebugText(
                f"Tilt  a:{np.rad2deg(alpha):+.1f}°  b:{np.rad2deg(beta):+.1f}°",
                [-0.22, -0.22, H0+0.10],
                textColorRGB=[0.6, 0.8, 1], textSize=1.1,
                replaceItemUniqueId=hud_tilt)

            ctrl_str = "[Arrows]=Push ball  [R]=Reset ball"
            hud_boost = p.addUserDebugText(
                ctrl_str,
                [-0.22, -0.22, H0+0.06],
                textColorRGB=[0.8, 0.8, 0.8],
                textSize=1.0,
                replaceItemUniqueId=hud_boost)

        if step % 240 == 0:
            dist = np.sqrt(ex**2 + ey**2)
            print(f"  t={step//240}s  "
                  f"ball=({bx:+.4f},{by:+.4f})  "
                  f"dist={dist:.4f}m")

        step += 1

except KeyboardInterrupt:
    print("\nStopped by user.")

p.disconnect()

print("\n=======================================")
print(" Stewart Platform Sim VERIFIED        ")
print("=======================================")
