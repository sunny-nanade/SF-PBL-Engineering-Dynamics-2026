"""
DC Motor Robotic Arm — Pick & Place Simulation
===============================================
Simulates a 3-DOF robotic arm driven by DC motors using velocity control
(mimicking PWM signals). The arm picks up a small cube and places it at
a target location, then repeats.

This version also performs simple mathematical modelling and generates
plots for the arm joints after the simulation:
  • Joint angle tracking (target vs actual)
  • DC motor velocity response (commanded vs actual vs first-order model)
  • Torque-speed relationship for the base joint

Run with:  python dc_motor_simulation.py
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
MAX_VELOCITY = 3.0          # rad/s — DC motor max speed
MAX_FORCE = 30.0            # N·m  — torque clamp
GRIPPER_FORCE = 10.0
RAMP_STEPS = 60             # steps to ramp velocity (smooth accel)
DC_TIME_CONSTANT = 0.12     # s — first-order motor approximation
STALL_TORQUE = MAX_FORCE    # N·m
NO_LOAD_SPEED = MAX_VELOCITY

# Joint indices (must match URDF order)
JOINT_BASE = 0
JOINT_SHOULDER = 1
JOINT_ELBOW = 2
JOINT_LEFT_FINGER = 3
JOINT_RIGHT_FINGER = 4
ARM_JOINTS = [JOINT_BASE, JOINT_SHOULDER, JOINT_ELBOW]
JOINT_NAMES = ["Base", "Shoulder", "Elbow"]

# ── Waypoints for pick & place ────────────────────────────────────────────────
HOME_ANGLES = [0.0, 0.0, 0.0]
PICK_ANGLES = [0.0, 0.6, 0.8]
LIFT_ANGLES = [0.0, 0.2, 0.4]
MOVE_ANGLES = [1.5, 0.2, 0.4]
PLACE_ANGLES = [1.5, 0.6, 0.8]
RETRACT_ANGLES = [1.5, 0.0, 0.0]


def create_telemetry():
    return {
        "time": [],
        "target_angles": [[] for _ in ARM_JOINTS],
        "angles": [[] for _ in ARM_JOINTS],
        "commanded_velocity": [[] for _ in ARM_JOINTS],
        "velocity": [[] for _ in ARM_JOINTS],
        "torque": [[] for _ in ARM_JOINTS],
    }


def connect_sim():
    """Start physics server and configure."""
    physics_id = p.connect(SIM_MODE)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)
    p.setTimeStep(TIME_STEP)
    return physics_id


def load_scene(urdf_dir):
    """Load ground plane, robot arm, and a small cube to pick."""
    p.loadURDF("plane.urdf")
    robot = p.loadURDF(
        os.path.join(urdf_dir, "dc_motor_arm.urdf"),
        basePosition=[0, 0, 0],
        useFixedBase=True,
    )

    cube_start = [0.25, 0.0, 0.025]
    cube_id = p.loadURDF(
        "cube_small.urdf",
        basePosition=cube_start,
        globalScaling=0.5,
    )
    p.changeVisualShape(cube_id, -1, rgbaColor=[0.9, 0.15, 0.15, 1.0])
    return robot, cube_id


def record_state(robot, telemetry, target_angles, commanded_velocities):
    sample_index = len(telemetry["time"])
    telemetry["time"].append(sample_index * TIME_STEP)
    for i, jidx in enumerate(ARM_JOINTS):
        state = p.getJointState(robot, jidx)
        telemetry["target_angles"][i].append(target_angles[i])
        telemetry["angles"][i].append(state[0])
        telemetry["commanded_velocity"][i].append(commanded_velocities[i])
        telemetry["velocity"][i].append(state[1])
        telemetry["torque"][i].append(state[3])


def advance_simulation(robot, telemetry, target_angles, commanded_velocities, steps):
    for _ in range(steps):
        p.stepSimulation()
        record_state(robot, telemetry, target_angles, commanded_velocities)
        if SIM_MODE == p.GUI:
            time.sleep(TIME_STEP)


def ramp_velocity_to_target(robot, telemetry, joint_indices, target_angles, steps=RAMP_STEPS):
    """
    Gradually ramp joint velocities toward target angles,
    simulating DC motor acceleration behaviour.
    """
    for _ in range(steps):
        commanded_velocities = []
        current_targets = []
        for idx, target in zip(joint_indices, target_angles):
            current = p.getJointState(robot, idx)[0]
            error = target - current
            vel = max(-MAX_VELOCITY, min(MAX_VELOCITY, error * 5.0))
            p.setJointMotorControl2(
                robot,
                idx,
                controlMode=p.VELOCITY_CONTROL,
                targetVelocity=vel,
                force=MAX_FORCE,
            )
            commanded_velocities.append(vel)
            current_targets.append(target)
        advance_simulation(robot, telemetry, current_targets, commanded_velocities, 1)

    settle_steps = 60
    for _ in range(settle_steps):
        commanded_velocities = []
        current_targets = []
        for idx, target in zip(joint_indices, target_angles):
            current = p.getJointState(robot, idx)[0]
            error = target - current
            vel = max(-MAX_VELOCITY, min(MAX_VELOCITY, error * 5.0))
            p.setJointMotorControl2(
                robot,
                idx,
                controlMode=p.VELOCITY_CONTROL,
                targetVelocity=vel,
                force=MAX_FORCE,
            )
            commanded_velocities.append(vel)
            current_targets.append(target)
        advance_simulation(robot, telemetry, current_targets, commanded_velocities, 1)


def close_gripper(robot, telemetry):
    """Close the prismatic gripper fingers."""
    p.setJointMotorControl2(
        robot, JOINT_LEFT_FINGER, p.VELOCITY_CONTROL, targetVelocity=-0.5, force=GRIPPER_FORCE
    )
    p.setJointMotorControl2(
        robot, JOINT_RIGHT_FINGER, p.VELOCITY_CONTROL, targetVelocity=0.5, force=GRIPPER_FORCE
    )
    current_targets = [p.getJointState(robot, idx)[0] for idx in ARM_JOINTS]
    current_velocities = [0.0, 0.0, 0.0]
    advance_simulation(robot, telemetry, current_targets, current_velocities, 60)


def open_gripper(robot, telemetry):
    """Open the prismatic gripper fingers."""
    p.setJointMotorControl2(
        robot, JOINT_LEFT_FINGER, p.VELOCITY_CONTROL, targetVelocity=0.5, force=GRIPPER_FORCE
    )
    p.setJointMotorControl2(
        robot, JOINT_RIGHT_FINGER, p.VELOCITY_CONTROL, targetVelocity=-0.5, force=GRIPPER_FORCE
    )
    current_targets = [p.getJointState(robot, idx)[0] for idx in ARM_JOINTS]
    current_velocities = [0.0, 0.0, 0.0]
    advance_simulation(robot, telemetry, current_targets, current_velocities, 60)


def pick_and_place_cycle(robot, telemetry, cycle_num):
    """Execute one full pick-and-place cycle."""
    print(f"[Cycle {cycle_num}] Moving to PICK position...")
    ramp_velocity_to_target(robot, telemetry, ARM_JOINTS, PICK_ANGLES)

    print(f"[Cycle {cycle_num}] Closing gripper — picking object...")
    close_gripper(robot, telemetry)

    print(f"[Cycle {cycle_num}] Lifting object...")
    ramp_velocity_to_target(robot, telemetry, ARM_JOINTS, LIFT_ANGLES)

    print(f"[Cycle {cycle_num}] Rotating to PLACE location...")
    ramp_velocity_to_target(robot, telemetry, ARM_JOINTS, MOVE_ANGLES)

    print(f"[Cycle {cycle_num}] Lowering to PLACE position...")
    ramp_velocity_to_target(robot, telemetry, ARM_JOINTS, PLACE_ANGLES)

    print(f"[Cycle {cycle_num}] Opening gripper — placing object...")
    open_gripper(robot, telemetry)

    print(f"[Cycle {cycle_num}] Retracting arm...")
    ramp_velocity_to_target(robot, telemetry, ARM_JOINTS, RETRACT_ANGLES)

    print(f"[Cycle {cycle_num}] Returning HOME...")
    ramp_velocity_to_target(robot, telemetry, ARM_JOINTS, HOME_ANGLES)

    print(f"[Cycle {cycle_num}] Pick-and-place complete.\n")


def simulate_dc_velocity_model(commanded_velocity, tau=DC_TIME_CONSTANT):
    """First-order DC motor velocity model: tau*w_dot + w = u."""
    model_velocity = []
    current_velocity = 0.0
    for command in commanded_velocity:
        current_velocity += (TIME_STEP / tau) * (command - current_velocity)
        model_velocity.append(current_velocity)
    return np.array(model_velocity)


def generate_plots(telemetry, output_dir):
    time_axis = np.array(telemetry["time"])
    target_angles = [np.array(series) for series in telemetry["target_angles"]]
    actual_angles = [np.array(series) for series in telemetry["angles"]]
    commanded_velocity = [np.array(series) for series in telemetry["commanded_velocity"]]
    actual_velocity = [np.array(series) for series in telemetry["velocity"]]
    torque = [np.array(series) for series in telemetry["torque"]]

    fig, axes = plt.subplots(3, 1, figsize=(11, 10), sharex=True)
    for i, ax in enumerate(axes):
        ax.plot(time_axis, target_angles[i], "--", label="Target angle")
        ax.plot(time_axis, actual_angles[i], label="Actual angle")
        ax.set_ylabel(f"{JOINT_NAMES[i]} (rad)")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best")
    axes[-1].set_xlabel("Time (s)")
    fig.suptitle("DC Motor Joint Tracking")
    fig.tight_layout()
    tracking_path = os.path.join(output_dir, "dc_motor_joint_tracking.png")
    fig.savefig(tracking_path, dpi=160, bbox_inches="tight")

    base_model_velocity = simulate_dc_velocity_model(commanded_velocity[0])
    fig2, ax2 = plt.subplots(figsize=(11, 5))
    ax2.plot(time_axis, commanded_velocity[0], "--", label="Commanded velocity")
    ax2.plot(time_axis, actual_velocity[0], label="Actual velocity")
    ax2.plot(time_axis, base_model_velocity, label="First-order model")
    ax2.set_title("DC Motor Base Joint Velocity Response")
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Angular velocity (rad/s)")
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="best")
    velocity_path = os.path.join(output_dir, "dc_motor_velocity_response.png")
    fig2.savefig(velocity_path, dpi=160, bbox_inches="tight")

    speed_line = np.linspace(0.0, NO_LOAD_SPEED, 100)
    torque_line = STALL_TORQUE * (1.0 - speed_line / NO_LOAD_SPEED)
    fig3, ax3 = plt.subplots(figsize=(8, 5))
    ax3.plot(speed_line, torque_line, label="Ideal DC torque-speed model")
    ax3.scatter(np.abs(actual_velocity[0]), np.abs(torque[0]), s=10, alpha=0.35, label="PyBullet samples")
    ax3.set_title("Base Joint Torque-Speed Characteristic")
    ax3.set_xlabel("Speed magnitude (rad/s)")
    ax3.set_ylabel("Torque magnitude (N·m)")
    ax3.grid(True, alpha=0.3)
    ax3.legend(loc="best")
    torque_path = os.path.join(output_dir, "dc_motor_torque_speed.png")
    fig3.savefig(torque_path, dpi=160, bbox_inches="tight")

    plt.show()
    return tracking_path, velocity_path, torque_path


def print_summary(telemetry):
    angle_errors = []
    for i in range(len(ARM_JOINTS)):
        final_error = abs(telemetry["target_angles"][i][-1] - telemetry["angles"][i][-1])
        angle_errors.append(final_error)
    base_actual_velocity = np.array(telemetry["velocity"][0])
    base_command_velocity = np.array(telemetry["commanded_velocity"][0])
    rms_velocity_error = np.sqrt(np.mean((base_actual_velocity - base_command_velocity) ** 2))

    print("\n--- DC Motor Modelling Summary ---")
    for name, err in zip(JOINT_NAMES, angle_errors):
        print(f"  {name} final angle error: {math.degrees(err):.3f} deg")
    print(f"  Base joint RMS velocity tracking error: {rms_velocity_error:.4f} rad/s")
    print(f"  First-order model time constant used: {DC_TIME_CONSTANT:.3f} s")


def main():
    print("=" * 60)
    print("DC Motor Robotic Arm — Pick & Place Simulation")
    print("=" * 60)

    connect_sim()
    urdf_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(urdf_dir, "plots")
    os.makedirs(output_dir, exist_ok=True)

    telemetry = create_telemetry()
    robot, cube_id = load_scene(urdf_dir)

    open_gripper(robot, telemetry)

    for cycle in range(1, 3):
        pick_and_place_cycle(robot, telemetry, cycle)

    print("Simulation finished successfully.")
    plot_paths = generate_plots(telemetry, output_dir)
    print_summary(telemetry)
    print("\nSaved plots:")
    for path in plot_paths:
        print(f"  {path}")

    time.sleep(1)
    p.disconnect()


if __name__ == "__main__":
    main()
