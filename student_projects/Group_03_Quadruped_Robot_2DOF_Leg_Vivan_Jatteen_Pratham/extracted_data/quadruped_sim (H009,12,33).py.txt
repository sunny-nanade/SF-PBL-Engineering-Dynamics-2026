"""
Stable Quadruped Walk - Laikago ZUP
=====================================
Fixed Forward Gait + Step Counter + Stop-and-Balance Phase
"""

import pybullet as p
import pybullet_data
import math
import time
import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# CONFIG
# ============================================================
GAIT         = "TROT"
SIM_TIME     = 60.0        # generous upper bound; we exit early via step counter

# Standing joint angles
HIP_STAND   =  0.0
THIGH_STAND =  0.70
KNEE_STAND  = -1.40

# Gait amplitudes
THIGH_AMP   =  0.25
KNEE_AMP    =  0.35

# Gait speed
GAIT_SPEED  =  2.0

# ── How many full trot cycles to walk before stopping ──────
WALK_CYCLES =  3           # 3 complete gait cycles ≈ ~3 visible forward strides

# POSITION CONTROL gains
KP          =  0.05
KD          =  0.5
FORCE       =  80.0

# Balance correction gains
KP_PITCH    =  0.2
KP_ROLL     =  0.1

# Physics
FRICTION    =  1.8
DAMPING     =  0.02
SPAWN_Z     =  0.34
DT          =  1.0 / 240.0

# Energy logging
BODY_MASS_EST      = 22.0
INERTIA_FACTOR_EST =  0.10

# ============================================================
# PYBULLET SETUP
# ============================================================
p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.81)
p.setPhysicsEngineParameter(
    fixedTimeStep=DT,
    numSolverIterations=50,
    numSubSteps=4)
p.setRealTimeSimulation(0)
p.resetDebugVisualizerCamera(1.4, 50, -20, [0, 0, 0.4])

plane = p.loadURDF("plane.urdf")
p.changeDynamics(plane, -1, lateralFriction=FRICTION, restitution=0.0)

# ============================================================
# LOAD ROBOT
# ============================================================
robot = p.loadURDF(
    "laikago/laikago_toes_zup.urdf",
    basePosition=[0, 0, SPAWN_Z],
    baseOrientation=[0, 0, 0, 1],
    useFixedBase=False,
    flags=p.URDF_USE_SELF_COLLISION_EXCLUDE_ALL_PARENTS)

for i in range(-1, p.getNumJoints(robot)):
    p.changeDynamics(robot, i,
                     lateralFriction=FRICTION,
                     restitution=0.0,
                     linearDamping=DAMPING,
                     angularDamping=DAMPING)

# ============================================================
# JOINT MAP
# ============================================================
jid = {}
for i in range(p.getNumJoints(robot)):
    info = p.getJointInfo(robot, i)
    jid[info[1].decode()] = i
    p.enableJointForceTorqueSensor(robot, i, enableSensor=1)

LEGS = {
    "FL": {
        "hip":   "FL_hip_motor_2_chassis_joint",
        "thigh": "FL_upper_leg_2_hip_motor_joint",
        "knee":  "FL_lower_leg_2_upper_leg_joint",
    },
    "FR": {
        "hip":   "FR_hip_motor_2_chassis_joint",
        "thigh": "FR_upper_leg_2_hip_motor_joint",
        "knee":  "FR_lower_leg_2_upper_leg_joint",
    },
    "RL": {
        "hip":   "RL_hip_motor_2_chassis_joint",
        "thigh": "RL_upper_leg_2_hip_motor_joint",
        "knee":  "RL_lower_leg_2_upper_leg_joint",
    },
    "RR": {
        "hip":   "RR_hip_motor_2_chassis_joint",
        "thigh": "RR_upper_leg_2_hip_motor_joint",
        "knee":  "RR_lower_leg_2_upper_leg_joint",
    },
}

# Diagonal trot phases
TROT_PHASE = {
    "FL": 0.0,
    "RR": 0.0,
    "FR": math.pi,
    "RL": math.pi,
}

# ============================================================
# HARD RESET TO STANDING POSE
# ============================================================
print("Resetting joints to standing pose...")
for i in range(p.getNumJoints(robot)):
    p.setJointMotorControl2(robot, i, p.VELOCITY_CONTROL, force=0)

for leg in LEGS.values():
    p.resetJointState(robot, jid[leg["hip"]],   HIP_STAND)
    p.resetJointState(robot, jid[leg["thigh"]], THIGH_STAND)
    p.resetJointState(robot, jid[leg["knee"]],  KNEE_STAND)

# ============================================================
# HELPERS
# ============================================================
def drive(joint_name, target):
    p.setJointMotorControl2(
        robot, jid[joint_name],
        p.POSITION_CONTROL,
        targetPosition=target,
        positionGain=KP,
        velocityGain=KD,
        force=FORCE)

def balance_adjust():
    _, orn = p.getBasePositionAndOrientation(robot)
    roll, pitch, _ = p.getEulerFromQuaternion(orn)
    pitch_c = max(-0.30, min(0.30, KP_PITCH * pitch))
    roll_c  = max(-0.20, min(0.20, KP_ROLL  * roll))
    return {
        "FL":  pitch_c - roll_c,
        "FR":  pitch_c + roll_c,
        "RL": -pitch_c - roll_c,
        "RR": -pitch_c + roll_c,
    }

def follow_camera():
    pos, _ = p.getBasePositionAndOrientation(robot)
    cam = p.getDebugVisualizerCamera()
    p.resetDebugVisualizerCamera(cam[10], cam[8], cam[9],
                                 [pos[0], pos[1], pos[2]])

def hold_stand(steps=720, label=""):
    """Hold standing pose with balance correction for `steps` sim steps."""
    for s in range(steps):
        adj = balance_adjust()
        for name, leg in LEGS.items():
            drive(leg["hip"],   HIP_STAND)
            drive(leg["thigh"], THIGH_STAND + adj[name])
            drive(leg["knee"],  KNEE_STAND  + adj[name])
        p.stepSimulation()
        log_state(t_global[0])
        follow_camera()
        t_global[0] += DT
        time.sleep(DT * 0.5)
    if label:
        print(label)

# ============================================================
# DATA LOGGING
# ============================================================
log_t, log_z, log_roll, log_pitch            = [], [], [], []
log_energy_t, log_energy_v, log_energy_total = [], [], []
log_fr_thigh,      log_fr_knee               = [], []
log_fr_thigh_tgt,  log_fr_knee_tgt           = [], []
log_fr_thigh_torque, log_fr_knee_torque      = [], []
log_phase_marker                             = []   # times when a new step starts

fr_thigh_id = jid[LEGS["FR"]["thigh"]]
fr_knee_id  = jid[LEGS["FR"]["knee"]]

current_thigh_tgt = THIGH_STAND
current_knee_tgt  = KNEE_STAND
t_global = [0.0]   # mutable so hold_stand() can update it

def log_state(t_now, phase_event=False):
    pos, orn         = p.getBasePositionAndOrientation(robot)
    lin_vel, ang_vel = p.getBaseVelocity(robot)
    roll, pitch, _   = p.getEulerFromQuaternion(orn)

    trans_ke = 0.5 * BODY_MASS_EST * sum(v*v for v in lin_vel)
    rot_ke   = 0.5 * (INERTIA_FACTOR_EST * BODY_MASS_EST) * sum(w*w for w in ang_vel)
    pe       = BODY_MASS_EST * 9.81 * pos[2]

    log_t.append(t_now)
    log_z.append(pos[2])
    log_roll.append(math.degrees(roll))
    log_pitch.append(math.degrees(pitch))
    log_energy_t.append(trans_ke + rot_ke)
    log_energy_v.append(pe)
    log_energy_total.append(trans_ke + rot_ke + pe)

    fr_thigh_state = p.getJointState(robot, fr_thigh_id)
    fr_knee_state  = p.getJointState(robot, fr_knee_id)

    log_fr_thigh.append(fr_thigh_state[0])
    log_fr_knee.append(fr_knee_state[0])
    log_fr_thigh_tgt.append(current_thigh_tgt)
    log_fr_knee_tgt.append(current_knee_tgt)
    log_fr_thigh_torque.append(fr_thigh_state[3])
    log_fr_knee_torque.append(fr_knee_state[3])

    if phase_event:
        log_phase_marker.append(t_now)

# ============================================================
# PHASE 1 — SETTLE
# ============================================================
print("Settling on legs...")
t = t_global[0]

for _ in range(800):
    adj = balance_adjust()
    for name, leg in LEGS.items():
        drive(leg["hip"],   HIP_STAND)
        drive(leg["thigh"], THIGH_STAND + adj[name])
        drive(leg["knee"],  KNEE_STAND  + adj[name])
    p.stepSimulation()
    log_state(t)
    follow_camera()
    t += DT
    time.sleep(DT * 0.5)

t_global[0] = t
pos, _ = p.getBasePositionAndOrientation(robot)
print(f"Settled at z = {pos[2]:.4f} m")
settle_end_t = t

# ============================================================
# SLIDERS
# ============================================================
sl_speed  = p.addUserDebugParameter("Speed scale (0=stop)",  0.0, 1.0, 0.8)
sl_gspeed = p.addUserDebugParameter("Gait speed (rad/s)",    0.2, 5.0, GAIT_SPEED)
sl_thamp  = p.addUserDebugParameter("Thigh amplitude (rad)", 0.01, 0.50, THIGH_AMP)
sl_kamp   = p.addUserDebugParameter("Knee lift (rad)",       0.01, 0.50, KNEE_AMP)

# ============================================================
# PHASE 2 — TROT WALK  (stops after WALK_CYCLES full cycles)
# ============================================================
print(f"\nWalking for {WALK_CYCLES} gait cycles...")
walk_start   = t_global[0]
status_t     = t_global[0]
cycles_done  = 0
prev_raw_phase = None     # track phase of the FL leg to count full cycles

walk_end_t   = None       # recorded when walk finishes

try:
    while t_global[0] < SIM_TIME:
        t = t_global[0]
        speed  = p.readUserDebugParameter(sl_speed)
        gspeed = p.readUserDebugParameter(sl_gspeed)
        t_amp  = p.readUserDebugParameter(sl_thamp)
        k_amp  = p.readUserDebugParameter(sl_kamp)

        ramp     = min(1.0, (t - walk_start) / 8.0)
        ramp     = ramp * ramp * (3.0 - 2.0 * ramp)   # smooth-step
        t_walk   = t - walk_start

        # ── Count complete gait cycles via FL phase wrap ─────
        fl_raw_phase = gspeed * t_walk    # FL phase (no offset)
        fl_cycle     = fl_raw_phase / (2.0 * math.pi)

        if prev_raw_phase is not None:
            prev_cycle = prev_raw_phase / (2.0 * math.pi)
            if int(fl_cycle) > int(prev_cycle):
                cycles_done += 1
                print(f"  Step cycle {cycles_done}/{WALK_CYCLES} at t={t:.2f}s")
                if cycles_done >= WALK_CYCLES:
                    walk_end_t = t
                    break
        prev_raw_phase = fl_raw_phase

        adj = balance_adjust()

        for leg_name, leg in LEGS.items():
            phase = gspeed * t_walk + TROT_PHASE[leg_name]

            # ── FORWARD GAIT FIX ──────────────────────────────────────────
            # Negate cosine so the thigh swings backward during stance phase,
            # pushing the body *forward* instead of backward.
            # Stance ≈ sin(phase) ≤ 0  →  cos goes from +1 → -1 (fwd → back)
            # Swing  ≈ sin(phase) >  0  →  cos goes from -1 → +1 (back → fwd)
            thigh_tgt = (THIGH_STAND
                         + adj[leg_name]
                         - speed * t_amp * ramp * math.cos(phase))   # <-- negated

            knee_tgt  = (KNEE_STAND
                         + adj[leg_name]
                         - k_amp * ramp * max(0.0, math.sin(phase)))

            drive(leg["hip"],   HIP_STAND)
            drive(leg["thigh"], thigh_tgt)
            drive(leg["knee"],  knee_tgt)

            if leg_name == "FR":
                current_thigh_tgt = thigh_tgt
                current_knee_tgt  = knee_tgt

        p.stepSimulation()
        # mark a phase event at each new cycle for the plot
        is_event = (prev_raw_phase is not None and
                    int(fl_cycle) > int(prev_raw_phase / (2*math.pi) - 1e-9))
        log_state(t, phase_event=False)
        follow_camera()
        t_global[0] += DT
        time.sleep(DT * 0.7)

        if t - status_t >= 1.5:
            pos2, orn2 = p.getBasePositionAndOrientation(robot)
            r2, pi2, _ = p.getEulerFromQuaternion(orn2)
            print(f"  t={t:5.2f}s  x={pos2[0]:+.3f}  y={pos2[1]:+.3f}  "
                  f"z={pos2[2]:+.3f}  roll={math.degrees(r2):+.1f}°  pitch={math.degrees(pi2):+.1f}°")
            status_t = t

except p.error:
    print("Window closed early.")
    p.disconnect()
    exit()

if walk_end_t is None:
    walk_end_t = t_global[0]

# ============================================================
# PHASE 3 — SMOOTH STOP (interpolate joints → standing, 1 s)
# ============================================================
print("\nSmooth stop — returning to standing pose...")
STOP_STEPS   = 240   # 1 second at 240 Hz

# Capture current joint positions as stop-ramp start
stop_start_thigh = {}
stop_start_knee  = {}
for leg_name, leg in LEGS.items():
    stop_start_thigh[leg_name] = p.getJointState(robot, jid[leg["thigh"]])[0]
    stop_start_knee[leg_name]  = p.getJointState(robot, jid[leg["knee"]])[0]

for s in range(STOP_STEPS):
    alpha = s / STOP_STEPS               # 0 → 1
    alpha = alpha * alpha * (3 - 2 * alpha)  # smooth-step ease

    adj = balance_adjust()
    for leg_name, leg in LEGS.items():
        thigh_tgt = (stop_start_thigh[leg_name] * (1 - alpha)
                     + (THIGH_STAND + adj[leg_name]) * alpha)
        knee_tgt  = (stop_start_knee[leg_name]  * (1 - alpha)
                     + (KNEE_STAND  + adj[leg_name]) * alpha)

        drive(leg["hip"],   HIP_STAND)
        drive(leg["thigh"], thigh_tgt)
        drive(leg["knee"],  knee_tgt)

        if leg_name == "FR":
            current_thigh_tgt = thigh_tgt
            current_knee_tgt  = knee_tgt

    p.stepSimulation()
    log_state(t_global[0])
    follow_camera()
    t_global[0] += DT
    time.sleep(DT * 0.5)

stop_end_t = t_global[0]

# ============================================================
# PHASE 4 — BALANCE HOLD (stand still, 4 s)
# ============================================================
print("Balancing on legs (4 s)...")
HOLD_STEPS = int(4.0 / DT)   # 4 seconds
hold_stand(HOLD_STEPS, label="Balance-hold complete.")
balance_end_t = t_global[0]

pos_final, _ = p.getBasePositionAndOrientation(robot)
print(f"\nFinal position: x={pos_final[0]:+.3f}  y={pos_final[1]:+.3f}  z={pos_final[2]:+.3f}")

p.disconnect()
print("Simulation complete — generating plots...")

# ============================================================
# ANALYSIS PLOTS
# ============================================================
if log_t:
    arr_t        = np.array(log_t)
    arr_z        = np.array(log_z)
    arr_roll     = np.array(log_roll)
    arr_pitch    = np.array(log_pitch)
    arr_ke       = np.array(log_energy_t)
    arr_pe       = np.array(log_energy_v)
    arr_e        = np.array(log_energy_total)
    arr_fr_thigh        = np.array(log_fr_thigh)
    arr_fr_knee         = np.array(log_fr_knee)
    arr_fr_thigh_tgt    = np.array(log_fr_thigh_tgt)
    arr_fr_knee_tgt     = np.array(log_fr_knee_tgt)
    arr_fr_thigh_torque = np.array(log_fr_thigh_torque)
    arr_fr_knee_torque  = np.array(log_fr_knee_torque)

    fig, axes = plt.subplots(4, 1, figsize=(14, 15))
    fig.suptitle(
        f"Quadruped Walk Analysis  |  KP={KP}  KD={KD}  FORCE={FORCE}\n"
        f"Walk {WALK_CYCLES} Cycles → Smooth Stop → 4 s Balance Hold",
        fontsize=13, fontweight="bold", y=0.99)

    # ── Shared phase-band helper ──────────────────────────────
    def add_bands(ax):
        ax.axvspan(0,             settle_end_t,  alpha=0.06, color="steelblue",  label="Settle")
        ax.axvspan(settle_end_t,  walk_end_t,    alpha=0.06, color="limegreen",  label="Walk")
        ax.axvspan(walk_end_t,    stop_end_t,    alpha=0.06, color="orange",     label="Stop ramp")
        ax.axvspan(stop_end_t,    balance_end_t, alpha=0.06, color="orchid",     label="Balance hold")
        for pm in log_phase_marker:
            ax.axvline(pm, color="gray", ls=":", lw=0.8, alpha=0.6)

    # ── Plot 1: Body Stability ────────────────────────────────
    ax1 = axes[0]
    ax1.plot(arr_t, arr_z,     color="royalblue",  lw=1.8, label="Body height z (m)")
    ax1.plot(arr_t, arr_roll,  color="darkorange", lw=1.2, label="Roll (deg)")
    ax1.plot(arr_t, arr_pitch, color="crimson",    lw=1.2, label="Pitch (deg)")
    add_bands(ax1)
    ax1.set_title("Plot 1 — Body Stability (Height · Roll · Pitch)")
    ax1.legend(loc="upper right", fontsize=8, ncol=2)
    ax1.grid(True, alpha=0.3)

    # ── Plot 2: Target Tracking ───────────────────────────────
    ax2 = axes[1]
    ax2.plot(arr_t, arr_fr_thigh,     color="forestgreen",  lw=1.6, label="FR Thigh Actual")
    ax2.plot(arr_t, arr_fr_thigh_tgt, color="lightgreen",   lw=1.4, ls="--", label="FR Thigh Target")
    ax2.plot(arr_t, arr_fr_knee,      color="mediumpurple", lw=1.6, label="FR Knee Actual")
    ax2.plot(arr_t, arr_fr_knee_tgt,  color="thistle",      lw=1.4, ls="--", label="FR Knee Target")
    add_bands(ax2)
    ax2.set_title("Plot 2 — Joint Target Tracking (FR leg)")
    ax2.legend(loc="upper right", fontsize=8, ncol=2)
    ax2.grid(True, alpha=0.3)

    # ── Plot 3: Motor Torques ─────────────────────────────────
    ax3 = axes[2]
    ax3.plot(arr_t, arr_fr_thigh_torque, color="forestgreen",  lw=1.2, label="FR Thigh Torque (N·m)")
    ax3.plot(arr_t, arr_fr_knee_torque,  color="mediumpurple", lw=1.2, label="FR Knee Torque (N·m)")
    ax3.axhline( FORCE, color="red", ls="--", lw=0.9, label="Force limit")
    ax3.axhline(-FORCE, color="red", ls="--", lw=0.9)
    add_bands(ax3)
    ax3.set_title(f"Plot 3 — Motor Torques (cap ±{FORCE} N·m)")
    ax3.set_ylabel("Torque (N·m)")
    ax3.legend(loc="upper right", fontsize=8, ncol=2)
    ax3.grid(True, alpha=0.3)

    # ── Plot 4: Energy ────────────────────────────────────────
    ax4 = axes[3]
    ax4.plot(arr_t, arr_ke, color="teal",      lw=1.4, label="Kinetic  T (J)")
    ax4.plot(arr_t, arr_pe, color="goldenrod", lw=1.4, label="Potential V (J)")
    ax4.plot(arr_t, arr_e,  color="black",     lw=2.0, label="Total  E = T+V (J)")
    add_bands(ax4)
    ax4.set_title("Plot 4 — Energy")
    ax4.set_xlabel("Time (s)")
    ax4.set_ylabel("Energy (J)")
    ax4.legend(loc="upper right", fontsize=8, ncol=2)
    ax4.grid(True, alpha=0.3)

    # Shared legend for the phase bands (once, at bottom)
    from matplotlib.patches import Patch
    band_legend = [
        Patch(color="steelblue", alpha=0.3, label="Settle"),
        Patch(color="limegreen", alpha=0.3, label="Walk"),
        Patch(color="orange",    alpha=0.3, label="Stop ramp"),
        Patch(color="orchid",    alpha=0.3, label="Balance hold"),
    ]
    fig.legend(handles=band_legend, loc="lower center", ncol=4,
               fontsize=9, framealpha=0.9, title="Simulation phases")

    plt.tight_layout(rect=[0, 0.04, 1, 0.98])
    out_path = "quadruped_debug_analysis.png"
    plt.savefig(out_path, dpi=150)
    print(f"Saved: {out_path}")
    plt.show()