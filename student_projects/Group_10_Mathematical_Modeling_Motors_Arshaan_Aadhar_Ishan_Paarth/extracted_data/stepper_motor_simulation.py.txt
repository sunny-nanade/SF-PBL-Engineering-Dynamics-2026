"""
Stepper Motor Robotic Arm — Precision Positioning Simulation
============================================================
Simulates a 3-DOF robotic arm driven by stepper motors using discrete
angular steps. The arm visits multiple precise waypoints and places
objects on an imaginary grid.

This version also performs simple mathematical modelling and generates
plots for the arm joints after the simulation:
  • Commanded staircase position vs actual position
  • Step count accumulation over time
  • Quantization error introduced by step resolution

Run with:  python stepper_motor_simulation.py
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
STEP_SIZE = math.radians(1.8)
STEPS_PER_MOVE = 8
HOLD_FORCE = 50.0
POSITION_GAIN = 0.8
GRIPPER_FORCE = 10.0

# Joint indices
JOINT_BASE = 0
JOINT_SHOULDER = 1
JOINT_ELBOW = 2
JOINT_LEFT_FINGER = 3
JOINT_RIGHT_FINGER = 4
ARM_JOINTS = [JOINT_BASE, JOINT_SHOULDER, JOINT_ELBOW]
JOINT_NAMES = ["Base", "Shoulder", "Elbow"]

# Precision grid waypoints (base, shoulder, elbow)
WAYPOINTS = [
    [0.0, 0.5, 0.7],
    [0.0, 0.2, 0.3],
    [0.5, 0.2, 0.3],
    [0.5, 0.5, 0.7],
    [0.5, 0.2, 0.3],
    [1.0, 0.2, 0.3],
    [1.0, 0.5, 0.7],
    [1.0, 0.2, 0.3],
    [0.0, 0.2, 0.3],
]


def create_telemetry():
    return {
        "time": [],
        "target_angles": [[] for _ in ARM_JOINTS],
        "angles": [[] for _ in ARM_JOINTS],
        "step_index": [[] for _ in ARM_JOINTS],
        "quantization_error": [[] for _ in ARM_JOINTS],
    }


def connect_sim():
    physics_id = p.connect(SIM_MODE)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)
    p.setTimeStep(TIME_STEP)
    return physics_id


def load_scene(urdf_dir):
    p.loadURDF("plane.urdf")
    robot = p.loadURDF(
        os.path.join(urdf_dir, "stepper_motor_arm.urdf"),
        basePosition=[0, 0, 0],
        useFixedBase=True,
    )

    cubes = []
    colors = [
        [0.2, 0.8, 0.2, 1.0],
        [0.2, 0.2, 0.9, 1.0],
    ]
    for i, color in enumerate(colors):
        cid = p.loadURDF(
            "cube_small.urdf",
            basePosition=[0.25 + i * 0.08, 0.0, 0.025],
            globalScaling=0.5,
        )
        p.changeVisualShape(cid, -1, rgbaColor=color)
        cubes.append(cid)
    return robot, cubes


def quantize_angle(current, target, step_size=STEP_SIZE):
    """Round an angle to the nearest full step toward target."""
    diff = target - current
    num_steps = round(diff / step_size)
    return current + num_steps * step_size


def record_state(robot, telemetry, target_angles, quantization_errors):
    sample_index = len(telemetry["time"])
    telemetry["time"].append(sample_index * TIME_STEP)
    for i, jidx in enumerate(ARM_JOINTS):
        state = p.getJointState(robot, jidx)
        telemetry["target_angles"][i].append(target_angles[i])
        telemetry["angles"][i].append(state[0])
        telemetry["step_index"][i].append(target_angles[i] / STEP_SIZE)
        telemetry["quantization_error"][i].append(quantization_errors[i])


def advance_simulation(robot, telemetry, target_angles, quantization_errors, steps):
    for _ in range(steps):
        p.stepSimulation()
        record_state(robot, telemetry, target_angles, quantization_errors)
        if SIM_MODE == p.GUI:
            time.sleep(TIME_STEP)


def step_to_angle(robot, telemetry, joint_idx, target_angle, current_targets, quantization_errors):
    """
    Move a single joint in discrete stepper increments toward target.
    """
    current = p.getJointState(robot, joint_idx)[0]
    diff = target_angle - current
    if abs(diff) < 1e-9:
        return

    num_steps = int(abs(diff) / STEP_SIZE) + 1
    direction = 1.0 if diff > 0 else -1.0
    joint_pos = ARM_JOINTS.index(joint_idx)

    for s in range(num_steps):
        next_pos = current + direction * STEP_SIZE * (s + 1)
        if direction > 0:
            next_pos = min(next_pos, target_angle)
        else:
            next_pos = max(next_pos, target_angle)

        current_targets[joint_pos] = next_pos
        p.setJointMotorControl2(
            robot,
            joint_idx,
            controlMode=p.POSITION_CONTROL,
            targetPosition=next_pos,
            force=HOLD_FORCE,
            positionGain=POSITION_GAIN,
        )
        advance_simulation(robot, telemetry, current_targets[:], quantization_errors, STEPS_PER_MOVE)


def step_joints_to(robot, telemetry, target_angles, quantization_errors):
    """Step all arm joints to their targets sequentially."""
    current_targets = [p.getJointState(robot, idx)[0] for idx in ARM_JOINTS]
    for jidx, tgt in zip(ARM_JOINTS, target_angles):
        step_to_angle(robot, telemetry, jidx, tgt, current_targets, quantization_errors)


def close_gripper(robot, telemetry):
    p.setJointMotorControl2(
        robot, JOINT_LEFT_FINGER, p.POSITION_CONTROL, targetPosition=-0.025, force=GRIPPER_FORCE
    )
    p.setJointMotorControl2(
        robot, JOINT_RIGHT_FINGER, p.POSITION_CONTROL, targetPosition=0.025, force=GRIPPER_FORCE
    )
    current_targets = [p.getJointState(robot, idx)[0] for idx in ARM_JOINTS]
    advance_simulation(robot, telemetry, current_targets, [0.0, 0.0, 0.0], 60)


def open_gripper(robot, telemetry):
    p.setJointMotorControl2(
        robot, JOINT_LEFT_FINGER, p.POSITION_CONTROL, targetPosition=0.0, force=GRIPPER_FORCE
    )
    p.setJointMotorControl2(
        robot, JOINT_RIGHT_FINGER, p.POSITION_CONTROL, targetPosition=0.0, force=GRIPPER_FORCE
    )
    current_targets = [p.getJointState(robot, idx)[0] for idx in ARM_JOINTS]
    advance_simulation(robot, telemetry, current_targets, [0.0, 0.0, 0.0], 60)


def generate_plots(telemetry, output_dir):
    time_axis = np.array(telemetry["time"])
    target_angles = [np.array(series) for series in telemetry["target_angles"]]
    actual_angles = [np.array(series) for series in telemetry["angles"]]
    step_index = [np.array(series) for series in telemetry["step_index"]]
    quantization_error = [np.array(series) for series in telemetry["quantization_error"]]

    fig, axes = plt.subplots(3, 1, figsize=(11, 10), sharex=True)
    for i, ax in enumerate(axes):
        ax.step(time_axis, target_angles[i], where="post", label="Ideal step command")
        ax.plot(time_axis, actual_angles[i], label="PyBullet response")
        ax.set_ylabel(f"{JOINT_NAMES[i]} (rad)")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best")
    axes[-1].set_xlabel("Time (s)")
    fig.suptitle("Stepper Position Tracking")
    fig.tight_layout()
    tracking_path = os.path.join(output_dir, "stepper_motor_position_tracking.png")
    fig.savefig(tracking_path, dpi=160, bbox_inches="tight")

    fig2, axes2 = plt.subplots(3, 1, figsize=(11, 10), sharex=True)
    for i, ax in enumerate(axes2):
        ax.step(time_axis, step_index[i], where="post", label="Full-step count")
        ax.set_ylabel(f"{JOINT_NAMES[i]} (steps)")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best")
    axes2[-1].set_xlabel("Time (s)")
    fig2.suptitle("Stepper Pulse-to-Position Mathematical Model")
    fig2.tight_layout()
    step_path = os.path.join(output_dir, "stepper_motor_step_count.png")
    fig2.savefig(step_path, dpi=160, bbox_inches="tight")

    fig3, axes3 = plt.subplots(3, 1, figsize=(11, 10), sharex=True)
    for i, ax in enumerate(axes3):
        ax.plot(time_axis, np.degrees(quantization_error[i]), label="Quantization error")
        ax.set_ylabel(f"{JOINT_NAMES[i]} (deg)")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best")
    axes3[-1].set_xlabel("Time (s)")
    fig3.suptitle("Stepper Resolution Error")
    fig3.tight_layout()
    error_path = os.path.join(output_dir, "stepper_motor_quantization_error.png")
    fig3.savefig(error_path, dpi=160, bbox_inches="tight")

    plt.show()
    return tracking_path, step_path, error_path


def main():
    print("=" * 60)
    print("Stepper Motor Robotic Arm — Precision Positioning Simulation")
    print("=" * 60)

    connect_sim()
    urdf_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(urdf_dir, "plots")
    os.makedirs(output_dir, exist_ok=True)

    telemetry = create_telemetry()
    robot, cubes = load_scene(urdf_dir)

    open_gripper(robot, telemetry)

    for wp_idx, wp in enumerate(WAYPOINTS):
        current_angles = [p.getJointState(robot, j)[0] for j in range(3)]
        quantized = [quantize_angle(c, t) for c, t in zip(current_angles, wp)]
        quantization_errors = [q - w for q, w in zip(quantized, wp)]

        print(
            f"[Step {wp_idx + 1}/{len(WAYPOINTS)}] Moving to waypoint "
            f"({quantized[0]:.3f}, {quantized[1]:.3f}, {quantized[2]:.3f}) rad"
        )
        step_joints_to(robot, telemetry, quantized, quantization_errors)

        if wp_idx == 0:
            print("  -> Closing gripper (pick)")
            close_gripper(robot, telemetry)
        elif wp_idx in (3, 6):
            print("  -> Opening gripper (place)")
            open_gripper(robot, telemetry)
        elif wp_idx == 8:
            print("  -> Closing gripper (pick next)")
            close_gripper(robot, telemetry)

    print("\n--- Final Joint States ---")
    for j in range(3):
        state = p.getJointState(robot, j)
        steps_from_zero = state[0] / STEP_SIZE
        print(f"  Joint {j}: {math.degrees(state[0]):.2f} deg ({steps_from_zero:.1f} full steps from zero)")

    plot_paths = generate_plots(telemetry, output_dir)
    print("\n--- Stepper Modelling Summary ---")
    for i, name in enumerate(JOINT_NAMES):
        max_quant_error = np.max(np.abs(telemetry["quantization_error"][i]))
        print(f"  {name}: max quantization error = {math.degrees(max_quant_error):.3f} deg")

    print("\nSaved plots:")
    for path in plot_paths:
        print(f"  {path}")

    print("\nStepper motor simulation finished successfully.")
    time.sleep(1)
    p.disconnect()


if __name__ == "__main__":
    main()
