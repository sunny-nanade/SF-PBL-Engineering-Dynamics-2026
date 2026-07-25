"""
Servo Motor Robotic Arm — Color Sorting Simulation
==================================================
Simulates a 3-DOF robotic arm driven by servo motors using PD position
control. The arm identifies colored cubes and places each into the
matching colored bin.

This version also performs simple mathematical modelling and generates
plots for the arm joints after the simulation:
  • Target vs actual joint angle tracking
  • Servo PD model response vs PyBullet response
  • Joint error convergence with settling information

Run with:  python servo_motor_simulation.py
Change p.DIRECT → p.GUI for interactive 3D viewer.
"""

import os
import math
import time
import pybullet as p
import pybullet_data
import matplotlib.pyplot as plt
import numpy as np

# ── Configuration ─────────────────────────────────────────────────────────────
SIM_MODE = p.GUI  # Change to p.DIRECT for headless mode
TIME_STEP = 1 / 240
SERVO_KP = 0.6
SERVO_KD = 0.2
SERVO_MAX_FORCE = 40.0
SERVO_MAX_VELOCITY = 5.0
GRIPPER_FORCE = 10.0
SETTLE_STEPS = 120
SERVO_MODEL_INERTIA = 1.0

# Joint indices
JOINT_BASE = 0
JOINT_SHOULDER = 1
JOINT_ELBOW = 2
JOINT_LEFT_FINGER = 3
JOINT_RIGHT_FINGER = 4
ARM_JOINTS = [JOINT_BASE, JOINT_SHOULDER, JOINT_ELBOW]
JOINT_NAMES = ["Base", "Shoulder", "Elbow"]

# ── Color definitions ────────────────────────────────────────────────────────
COLORS = {
    "red": [0.9, 0.15, 0.15, 1.0],
    "green": [0.15, 0.8, 0.2, 1.0],
    "blue": [0.15, 0.2, 0.9, 1.0],
}

BIN_ANGLES = {
    "red": 0.5,
    "green": 1.57,
    "blue": 2.6,
}

CUBE_POSITIONS = [
    [0.25, 0.05, 0.025],
    [0.25, -0.05, 0.025],
    [0.25, 0.15, 0.025],
]
CUBE_COLORS = ["red", "green", "blue"]


def create_telemetry():
    return {
        "time": [],
        "target_angles": [[] for _ in ARM_JOINTS],
        "angles": [[] for _ in ARM_JOINTS],
        "velocity": [[] for _ in ARM_JOINTS],
        "error": [[] for _ in ARM_JOINTS],
    }


def connect_sim():
    physics_id = p.connect(SIM_MODE)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)
    p.setTimeStep(TIME_STEP)
    return physics_id


def load_scene(urdf_dir):
    """Load ground, robot, colored cubes, and sorting bins."""
    p.loadURDF("plane.urdf")
    robot = p.loadURDF(
        os.path.join(urdf_dir, "servo_motor_arm.urdf"),
        basePosition=[0, 0, 0],
        useFixedBase=True,
    )

    cubes = []
    cube_color_names = []
    for pos, color_name in zip(CUBE_POSITIONS, CUBE_COLORS):
        cid = p.loadURDF("cube_small.urdf", basePosition=pos, globalScaling=0.5)
        p.changeVisualShape(cid, -1, rgbaColor=COLORS[color_name])
        cubes.append(cid)
        cube_color_names.append(color_name)

    for color_name, base_angle in BIN_ANGLES.items():
        bx = 0.3 * math.cos(base_angle)
        by = 0.3 * math.sin(base_angle)
        bin_id = p.loadURDF("cube_small.urdf", basePosition=[bx, by, 0.005], globalScaling=1.0)
        rgba = list(COLORS[color_name])
        rgba[3] = 0.4
        p.changeVisualShape(bin_id, -1, rgbaColor=rgba)
        p.changeDynamics(bin_id, -1, mass=0)

    return robot, cubes, cube_color_names


def record_state(robot, telemetry, target_angles):
    sample_index = len(telemetry["time"])
    telemetry["time"].append(sample_index * TIME_STEP)
    for i, jidx in enumerate(ARM_JOINTS):
        state = p.getJointState(robot, jidx)
        telemetry["target_angles"][i].append(target_angles[i])
        telemetry["angles"][i].append(state[0])
        telemetry["velocity"][i].append(state[1])
        telemetry["error"][i].append(target_angles[i] - state[0])


def advance_simulation(robot, telemetry, target_angles, steps):
    for _ in range(steps):
        p.stepSimulation()
        record_state(robot, telemetry, target_angles)
        if SIM_MODE == p.GUI:
            time.sleep(TIME_STEP)


def servo_move(robot, telemetry, target_angles, steps=SETTLE_STEPS):
    """
    Command servos to target angles using PD position control.
    """
    for jidx, tgt in zip(ARM_JOINTS, target_angles):
        p.setJointMotorControl2(
            robot,
            jidx,
            controlMode=p.POSITION_CONTROL,
            targetPosition=tgt,
            force=SERVO_MAX_FORCE,
            maxVelocity=SERVO_MAX_VELOCITY,
            positionGain=SERVO_KP,
            velocityGain=SERVO_KD,
        )
    advance_simulation(robot, telemetry, target_angles, steps)


def close_gripper(robot, telemetry):
    p.setJointMotorControl2(
        robot, JOINT_LEFT_FINGER, p.POSITION_CONTROL, targetPosition=-0.025, force=GRIPPER_FORCE
    )
    p.setJointMotorControl2(
        robot, JOINT_RIGHT_FINGER, p.POSITION_CONTROL, targetPosition=0.025, force=GRIPPER_FORCE
    )
    current_targets = [p.getJointState(robot, idx)[0] for idx in ARM_JOINTS]
    advance_simulation(robot, telemetry, current_targets, 60)


def open_gripper(robot, telemetry):
    p.setJointMotorControl2(
        robot, JOINT_LEFT_FINGER, p.POSITION_CONTROL, targetPosition=0.0, force=GRIPPER_FORCE
    )
    p.setJointMotorControl2(
        robot, JOINT_RIGHT_FINGER, p.POSITION_CONTROL, targetPosition=0.0, force=GRIPPER_FORCE
    )
    current_targets = [p.getJointState(robot, idx)[0] for idx in ARM_JOINTS]
    advance_simulation(robot, telemetry, current_targets, 60)


def get_position_error(robot, target_angles):
    """Return max absolute position error across arm joints."""
    errors = []
    for jidx, tgt in zip(ARM_JOINTS, target_angles):
        actual = p.getJointState(robot, jidx)[0]
        errors.append(abs(actual - tgt))
    return max(errors)


def sort_cube(robot, telemetry, cube_idx, color_name):
    """Pick up a cube and place it in the correct color bin."""
    bin_angle = BIN_ANGLES[color_name]

    pick_reach = [0.15, 0.6, 0.8]
    pick_lift = [0.15, 0.2, 0.3]
    place_hover = [bin_angle, 0.2, 0.3]
    place_lower = [bin_angle, 0.6, 0.8]
    home = [1.57, 0.0, 0.0]

    print(f"  [Pick] Reaching for {color_name} cube #{cube_idx + 1}...")
    servo_move(robot, telemetry, pick_reach)
    close_gripper(robot, telemetry)
    print("  [Grip] Cube gripped.")

    print("  [Lift] Lifting cube...")
    servo_move(robot, telemetry, pick_lift)

    print(f"  [Move] Rotating to {color_name} bin (theta = {math.degrees(bin_angle):.0f} deg)...")
    servo_move(robot, telemetry, place_hover)

    print("  [Place] Lowering into bin...")
    servo_move(robot, telemetry, place_lower)
    open_gripper(robot, telemetry)
    print(f"  [Release] Cube placed in {color_name} bin.")

    error = get_position_error(robot, place_lower)
    print(f"  [Accuracy] Position error: {math.degrees(error):.3f} deg")

    servo_move(robot, telemetry, home)


def simulate_servo_pd_model(target_angles):
    """
    Simple second-order servo model:
    theta_dd = kp*(theta_ref - theta) - kd*theta_d
    """
    model_angles = [[] for _ in ARM_JOINTS]
    model_velocities = [0.0 for _ in ARM_JOINTS]
    model_positions = [0.0 for _ in ARM_JOINTS]

    for k in range(len(target_angles[0])):
        for i in range(len(ARM_JOINTS)):
            error = target_angles[i][k] - model_positions[i]
            acceleration = (
                SERVO_KP * error - SERVO_KD * model_velocities[i]
            ) / SERVO_MODEL_INERTIA
            model_velocities[i] += acceleration * TIME_STEP
            model_velocities[i] = max(-SERVO_MAX_VELOCITY, min(SERVO_MAX_VELOCITY, model_velocities[i]))
            model_positions[i] += model_velocities[i] * TIME_STEP
            model_angles[i].append(model_positions[i])
    return [np.array(series) for series in model_angles]


def compute_settling_time(time_axis, error_series, threshold_deg=2.0):
    threshold = math.radians(threshold_deg)
    abs_error = np.abs(error_series)
    for i in range(len(abs_error)):
        if np.all(abs_error[i:] <= threshold):
            return time_axis[i]
    return None


def generate_plots(telemetry, output_dir):
    time_axis = np.array(telemetry["time"])
    target_angles = [np.array(series) for series in telemetry["target_angles"]]
    actual_angles = [np.array(series) for series in telemetry["angles"]]
    errors = [np.array(series) for series in telemetry["error"]]
    velocities = [np.array(series) for series in telemetry["velocity"]]
    model_angles = simulate_servo_pd_model(target_angles)

    fig, axes = plt.subplots(3, 1, figsize=(11, 10), sharex=True)
    for i, ax in enumerate(axes):
        ax.plot(time_axis, target_angles[i], "--", label="Target")
        ax.plot(time_axis, actual_angles[i], label="PyBullet")
        ax.plot(time_axis, model_angles[i], label="PD model")
        ax.set_ylabel(f"{JOINT_NAMES[i]} (rad)")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best")
    axes[-1].set_xlabel("Time (s)")
    fig.suptitle("Servo Joint Tracking and Mathematical Model")
    fig.tight_layout()
    tracking_path = os.path.join(output_dir, "servo_motor_joint_tracking.png")
    fig.savefig(tracking_path, dpi=160, bbox_inches="tight")

    fig2, axes2 = plt.subplots(3, 1, figsize=(11, 10), sharex=True)
    for i, ax in enumerate(axes2):
        ax.plot(time_axis, np.degrees(errors[i]), label="Position error")
        settling_time = compute_settling_time(time_axis, errors[i])
        if settling_time is not None:
            ax.axvline(settling_time, color="tab:red", linestyle="--", alpha=0.7, label="2 deg settling")
        ax.set_ylabel(f"{JOINT_NAMES[i]} (deg)")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best")
    axes2[-1].set_xlabel("Time (s)")
    fig2.suptitle("Servo Position Error Convergence")
    fig2.tight_layout()
    error_path = os.path.join(output_dir, "servo_motor_error_convergence.png")
    fig2.savefig(error_path, dpi=160, bbox_inches="tight")

    fig3, axes3 = plt.subplots(3, 1, figsize=(11, 10), sharex=True)
    for i, ax in enumerate(axes3):
        ax.plot(time_axis, velocities[i], label="Angular velocity")
        ax.set_ylabel(f"{JOINT_NAMES[i]} (rad/s)")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best")
    axes3[-1].set_xlabel("Time (s)")
    fig3.suptitle("Servo Joint Velocity Response")
    fig3.tight_layout()
    velocity_path = os.path.join(output_dir, "servo_motor_velocity_response.png")
    fig3.savefig(velocity_path, dpi=160, bbox_inches="tight")

    plt.show()
    return tracking_path, error_path, velocity_path


def print_summary(telemetry):
    time_axis = np.array(telemetry["time"])
    errors = [np.array(series) for series in telemetry["error"]]
    print("\n--- Servo Modelling Summary ---")
    for name, series in zip(JOINT_NAMES, errors):
        final_error_deg = math.degrees(abs(series[-1]))
        settling_time = compute_settling_time(time_axis, series)
        settling_text = f"{settling_time:.3f} s" if settling_time is not None else "not settled within 2 deg"
        print(f"  {name}: final error = {final_error_deg:.3f} deg, settling time = {settling_text}")


def main():
    print("=" * 60)
    print("Servo Motor Robotic Arm — Color Sorting Simulation")
    print("=" * 60)

    connect_sim()
    urdf_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(urdf_dir, "plots")
    os.makedirs(output_dir, exist_ok=True)

    telemetry = create_telemetry()
    robot, cubes, cube_color_names = load_scene(urdf_dir)

    open_gripper(robot, telemetry)
    servo_move(robot, telemetry, [1.57, 0.0, 0.0])

    for idx, color_name in enumerate(cube_color_names):
        print(f"\n--- Sorting cube {idx + 1}/{len(cubes)}: {color_name.upper()} ---")
        sort_cube(robot, telemetry, idx, color_name)

    print("\n--- Final Joint States ---")
    for j in range(3):
        state = p.getJointState(robot, j)
        print(f"  Joint {j}: position = {math.degrees(state[0]):.2f} deg, velocity = {state[1]:.4f} rad/s")

    plot_paths = generate_plots(telemetry, output_dir)
    print_summary(telemetry)
    print("\nSaved plots:")
    for path in plot_paths:
        print(f"  {path}")

    print("\nServo motor sorting simulation finished successfully.")
    time.sleep(1)
    p.disconnect()


if __name__ == "__main__":
    main()
