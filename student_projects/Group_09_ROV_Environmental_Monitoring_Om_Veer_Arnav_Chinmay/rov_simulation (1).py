from __future__ import annotations

"""PyBullet underwater ROV simulator aimed at classroom-style validation.

This file is intentionally self-contained so a student team can trace how the
world, physics, control logic, sensors, and telemetry fit together.
"""

import argparse
import csv
import math
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, Optional

import matplotlib.pyplot as plt
import numpy as np
import pybullet as p
import pybullet_data


# Global constants are kept near the top so the simulation setup is easy to explain.
GRAVITY = 9.81
SPEED_OF_SOUND_WATER = 1500.0
SEABED_HIT_TOLERANCE_M = 0.08
SEABED_CELL_SIZE_M = 0.5
AXIS_ORDER = ("surge", "sway", "heave", "roll", "pitch", "yaw")
MANUAL_MODE = "MANUAL"
AUTOMATIC_MODE = "AUTOMATIC"
ESCAPE_KEY = 27
CAMERA_TOGGLE_KEY = ord("c")
STATIC_CAMERA_DISTANCE = 4.8
STATIC_CAMERA_YAW = 32.0
STATIC_CAMERA_PITCH = -22.0
TPP_CAMERA_OFFSET_LOCAL = np.array([-2.0, 0.0, 1.0], dtype=float)


@dataclass(frozen=True)
class CylinderGeometry:
    """Geometric and inertial properties of the cylindrical ROV hull."""

    radius: float = 0.2
    length: float = 0.6
    mass: float = 73.89
    com_offset_z: float = -0.035

    @property
    def volume(self) -> float:
        # Displaced water volume for a cylinder: V = pi * r^2 * L.
        return math.pi * self.radius**2 * self.length

    @property
    def inertia_xx(self) -> float:
        # Solid-cylinder inertia about the body x-axis (roll axis).
        return 0.5 * self.mass * self.radius**2

    @property
    def inertia_yy(self) -> float:
        # Solid-cylinder inertia about the body y/z axes (pitch/yaw axes).
        return (1.0 / 12.0) * self.mass * (3.0 * self.radius**2 + self.length**2)

    @property
    def inertia_zz(self) -> float:
        return self.inertia_yy

    @property
    def projected_areas(self) -> np.ndarray:
        # Effective areas used by the quadratic drag model.
        frontal_area = math.pi * self.radius**2
        lateral_area = 2.0 * self.radius * self.length
        return np.array([frontal_area, lateral_area, lateral_area], dtype=float)

    @property
    def angular_drag_area(self) -> np.ndarray:
        # Simple characteristic areas for rotational drag around each axis.
        roll_area = math.pi * self.radius**4
        pitch_yaw_area = self.length * self.radius**3
        return np.array([roll_area, pitch_yaw_area, pitch_yaw_area], dtype=float)


@dataclass
class HydroCoefficients:
    """Hydrodynamic constants that define buoyancy and drag behavior."""

    rho: float = 1000.0
    drag_coefficient: float = 0.95
    buoyancy_multiplier: float = 0.98
    center_of_buoyancy_local: np.ndarray = field(
        default_factory=lambda: np.array([0.0, 0.0, 0.015], dtype=float)
    )
    angular_drag_coefficient: np.ndarray = field(
        default_factory=lambda: np.array([6.0, 8.0, 8.0], dtype=float)
    )


@dataclass
class RuntimeTuning:
    """Live-tunable values read from the PyBullet side panel."""

    rho: float
    drag_coefficient: float
    buoyancy_multiplier: float
    sonar_max_range: float = 6.0
    num_lasers: int = 5
    detection_threshold: float = 2.0
    ping_frequency_hz: float = 10.0
    sonar_noise_level_m: float = 0.04
    zone_transition_smoothing_m: float = 2.0


@dataclass
class ROVParameters:
    """Top-level vehicle configuration used when spawning the ROV."""

    geometry: CylinderGeometry = field(default_factory=CylinderGeometry)
    hydro: HydroCoefficients = field(default_factory=HydroCoefficients)
    start_position: np.ndarray = field(
        default_factory=lambda: np.array([0.0, 0.0, -0.9], dtype=float)
    )
    start_rpy: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=float))


@dataclass(frozen=True)
class Thruster:
    """One idealized thruster with a fixed mounting point and direction."""

    name: str
    position_local: np.ndarray
    direction_local: np.ndarray
    max_force_newtons: float


@dataclass(frozen=True)
class GateTarget:
    """Center point and dimensions of one navigation gate."""

    center_world: np.ndarray
    yaw: float
    width: float
    height: float


@dataclass
class MissionStatus:
    """Mission progress and summary values written to the log files."""

    current_gate_index: int = 0
    gate_pass_times: list[float] = field(default_factory=list)
    impact_events: list[dict[str, object]] = field(default_factory=list)
    start_time_s: Optional[float] = None
    completion_time_s: Optional[float] = None
    total_energy_consumed_j: float = 0.0
    completed: bool = False
    goal_radius_m: float = 0.55
    start_radius_m: float = 1.00


@dataclass(frozen=True)
class PathGuidance:
    """Geometric guidance information for the active path segment."""

    active_waypoint_index: int
    active_waypoint_world: np.ndarray
    segment_start_world: np.ndarray
    segment_end_world: np.ndarray
    heading_vector_world: np.ndarray
    crosstrack_vector_world: np.ndarray
    crosstrack_error_m: float


@dataclass(frozen=True)
class TankBounds:
    """Axis-aligned limits of the test tank."""

    x_min: float
    x_max: float
    y_min: float
    y_max: float
    z_min: float
    z_max: float

    def clearances(self, position: np.ndarray) -> dict[str, float]:
        # Clearance values are useful both for autopilot logic and validation plots.
        return {
            "clearance_front_m": float(self.x_max - position[0]),
            "clearance_back_m": float(position[0] - self.x_min),
            "clearance_left_m": float(position[1] - self.y_min),
            "clearance_right_m": float(self.y_max - position[1]),
            "clearance_top_m": float(self.z_max - position[2]),
            "clearance_bottom_m": float(position[2] - self.z_min),
        }


SECTOR_DEFINITIONS = (
    (1, "Coastal", -math.inf, -15.0, [0.20, 0.78, 0.92, 0.05]),
    (2, "Transition", -15.0, 0.0, [0.16, 0.62, 0.82, 0.05]),
    (3, "Deep Sea", 0.0, 15.0, [0.10, 0.32, 0.66, 0.06]),
    (4, "Abyssal", 15.0, math.inf, [0.06, 0.12, 0.38, 0.07]),
)

SECTOR_ENVIRONMENTAL_BASES = {
    1: {"temperature_c": 25.0, "luminosity_lux": 1200.0, "oxygen_pct": 100.0, "ph": 8.15},
    2: {"temperature_c": 18.0, "luminosity_lux": 450.0, "oxygen_pct": 80.0, "ph": 7.95},
    3: {"temperature_c": 8.0, "luminosity_lux": 40.0, "oxygen_pct": 40.0, "ph": 7.75},
    4: {"temperature_c": 2.0, "luminosity_lux": 0.0, "oxygen_pct": 20.0, "ph": 7.55},
}


def get_environmental_sector(x_position: float) -> dict[str, object]:
    """Return the sector metadata for a given x-position."""
    for sector_index, sector_name, x_min, x_max, tint_rgba in SECTOR_DEFINITIONS:
        if x_min <= x_position < x_max:
            return {
                "sector_index": sector_index,
                "sector_name": sector_name,
                "sector_tint_rgba": np.array(tint_rgba, dtype=float),
            }
    last_sector = SECTOR_DEFINITIONS[-1]
    return {
        "sector_index": last_sector[0],
        "sector_name": last_sector[1],
        "sector_tint_rgba": np.array(last_sector[4], dtype=float),
    }


def get_environmental_data(
    position: np.ndarray,
    zone_transition_smoothing_m: float = 0.0,
) -> dict[str, object]:
    """Return the local environmental sample for the current sector and x-position."""
    position = np.array(position, dtype=float)
    sector = get_environmental_sector(float(position[0]))
    sector_index = int(sector["sector_index"])
    x_position = float(position[0])
    smoothing_m = max(float(zone_transition_smoothing_m), 0.0)

    def smoothstep(edge0: float, edge1: float, value: float) -> float:
        if edge1 <= edge0:
            return 1.0 if value >= edge1 else 0.0
        t = float(np.clip((value - edge0) / (edge1 - edge0), 0.0, 1.0))
        return t * t * (3.0 - 2.0 * t)

    zone_weights = {
        1: 1.0 if x_position < -15.0 else 0.0,
        2: 1.0 if -15.0 <= x_position < 0.0 else 0.0,
        3: 1.0 if 0.0 <= x_position < 15.0 else 0.0,
        4: 1.0 if x_position >= 15.0 else 0.0,
    }

    if smoothing_m > 1e-6:
        boundaries = [(-15.0, 1, 2), (0.0, 2, 3), (15.0, 3, 4)]
        for boundary_x, left_zone, right_zone in boundaries:
            if abs(x_position - boundary_x) <= smoothing_m:
                blend = smoothstep(boundary_x - smoothing_m, boundary_x + smoothing_m, x_position)
                zone_weights = {zone: 0.0 for zone in zone_weights}
                zone_weights[left_zone] = 1.0 - blend
                zone_weights[right_zone] = blend
                break

    temperature_c = sum(
        zone_weights[zone_index] * SECTOR_ENVIRONMENTAL_BASES[zone_index]["temperature_c"]
        for zone_index in zone_weights
    )
    luminosity_lux = sum(
        zone_weights[zone_index] * SECTOR_ENVIRONMENTAL_BASES[zone_index]["luminosity_lux"]
        for zone_index in zone_weights
    )
    oxygen_pct = sum(
        zone_weights[zone_index] * SECTOR_ENVIRONMENTAL_BASES[zone_index]["oxygen_pct"]
        for zone_index in zone_weights
    )
    ph = sum(
        zone_weights[zone_index] * SECTOR_ENVIRONMENTAL_BASES[zone_index]["ph"]
        for zone_index in zone_weights
    )

    return {
        "sector_index": sector_index,
        "sector_name": str(sector["sector_name"]),
        "temperature_c": float(temperature_c),
        "luminosity_lux": float(luminosity_lux),
        "oxygen_pct": float(oxygen_pct),
        "ph": float(ph),
    }


class ParameterPanel:
    """Interactive Phase 3 validation sliders."""

    def __init__(self, client_id: int, defaults: RuntimeTuning) -> None:
        self.client_id = client_id
        # Every slider maps directly to a physical or sensing parameter used later in the loop.
        self.slider_ids = {
            "rho": p.addUserDebugParameter(
                "Fluid_Density_Rho", 990.0, 1030.0, defaults.rho, physicsClientId=client_id
            ),
            "drag_coefficient": p.addUserDebugParameter(
                "Drag_Coefficient", 0.1, 2.0, defaults.drag_coefficient, physicsClientId=client_id
            ),
            "buoyancy_multiplier": p.addUserDebugParameter(
                "Buoyancy_Multiplier",
                0.80,
                1.20,
                defaults.buoyancy_multiplier,
                physicsClientId=client_id,
            ),
            "sonar_max_range": p.addUserDebugParameter(
                "Sonar_Max_Range", 5.0, 8.0, defaults.sonar_max_range, physicsClientId=client_id
            ),
            "num_lasers": p.addUserDebugParameter(
                "Num_Lasers", 3.0, 20.0, float(defaults.num_lasers), physicsClientId=client_id
            ),
            "detection_threshold": p.addUserDebugParameter(
                "Detection_Threshold",
                0.5,
                5.0,
                defaults.detection_threshold,
                physicsClientId=client_id,
            ),
            "ping_frequency_hz": p.addUserDebugParameter(
                "Ping_Frequency_Hz", 5.0, 30.0, defaults.ping_frequency_hz, physicsClientId=client_id
            ),
            "sonar_noise_level_m": p.addUserDebugParameter(
                "Sonar_Noise_Level",
                0.0,
                0.5,
                defaults.sonar_noise_level_m,
                physicsClientId=client_id,
            ),
            "zone_transition_smoothing_m": p.addUserDebugParameter(
                "Zone_Transition_Smoothing",
                0.0,
                6.0,
                defaults.zone_transition_smoothing_m,
                physicsClientId=client_id,
            ),
        }

    def read(self) -> RuntimeTuning:
        # Read raw GUI values first, then clamp them to keep the simulation in a safe range.
        sonar_max_range = float(
            p.readUserDebugParameter(self.slider_ids["sonar_max_range"], physicsClientId=self.client_id)
        )
        detection_threshold = float(
            p.readUserDebugParameter(
                self.slider_ids["detection_threshold"], physicsClientId=self.client_id
            )
        )
        ping_frequency_hz = float(
            p.readUserDebugParameter(self.slider_ids["ping_frequency_hz"], physicsClientId=self.client_id)
        )
        sonar_noise_level_m = float(
            p.readUserDebugParameter(self.slider_ids["sonar_noise_level_m"], physicsClientId=self.client_id)
        )
        zone_transition_smoothing_m = float(
            p.readUserDebugParameter(
                self.slider_ids["zone_transition_smoothing_m"], physicsClientId=self.client_id
            )
        )
        return RuntimeTuning(
            rho=float(p.readUserDebugParameter(self.slider_ids["rho"], physicsClientId=self.client_id)),
            drag_coefficient=float(
                p.readUserDebugParameter(
                    self.slider_ids["drag_coefficient"], physicsClientId=self.client_id
                )
            ),
            buoyancy_multiplier=float(
                p.readUserDebugParameter(
                    self.slider_ids["buoyancy_multiplier"], physicsClientId=self.client_id
                )
            ),
            sonar_max_range=float(np.clip(sonar_max_range, 5.0, 8.0)),
            num_lasers=int(
                round(
                    p.readUserDebugParameter(
                        self.slider_ids["num_lasers"], physicsClientId=self.client_id
                    )
                )
            ),
            detection_threshold=float(np.clip(detection_threshold, 0.5, sonar_max_range)),
            ping_frequency_hz=float(np.clip(ping_frequency_hz, 5.0, 30.0)),
            sonar_noise_level_m=float(np.clip(sonar_noise_level_m, 0.0, 0.5)),
            zone_transition_smoothing_m=float(np.clip(zone_transition_smoothing_m, 0.0, 6.0)),
        )


class ROVController:
    """6-thruster mixer for surge, sway, heave, and yaw authority."""

    def __init__(self) -> None:
        # The horizontal thrusters are diagonal so they can share surge, sway, and yaw work.
        root_half2 = math.sqrt(0.5)
        self.thrusters = [
            Thruster(
                "front_port",
                np.array([0.22, 0.18, 0.0], dtype=float),
                np.array([root_half2, -root_half2, 0.0], dtype=float),
                56.0,
            ),
            Thruster(
                "front_starboard",
                np.array([0.22, -0.18, 0.0], dtype=float),
                np.array([root_half2, root_half2, 0.0], dtype=float),
                56.0,
            ),
            Thruster(
                "rear_port",
                np.array([-0.22, 0.18, 0.0], dtype=float),
                np.array([root_half2, root_half2, 0.0], dtype=float),
                56.0,
            ),
            Thruster(
                "rear_starboard",
                np.array([-0.22, -0.18, 0.0], dtype=float),
                np.array([root_half2, -root_half2, 0.0], dtype=float),
                56.0,
            ),
            Thruster(
                "vertical_fore",
                np.array([0.0, 0.16, 0.0], dtype=float),
                np.array([0.0, 0.0, 1.0], dtype=float),
                44.0,
            ),
            Thruster(
                "vertical_aft",
                np.array([0.0, -0.16, 0.0], dtype=float),
                np.array([0.0, 0.0, 1.0], dtype=float),
                44.0,
            ),
        ]
        self.axis_weights = np.array([155.0, 130.0, 120.0, 14.0, 12.0, 44.0], dtype=float)
        self.command_vector = np.zeros(len(self.thrusters), dtype=float)
        self.pwm_vector = np.full(len(self.thrusters), 1500.0, dtype=float)
        self._wrench_matrix = self._build_wrench_matrix()
        self._mixer = np.linalg.pinv(self._wrench_matrix)

    def _build_wrench_matrix(self) -> np.ndarray:
        """Map each thruster command to the force/torque it can create on the body."""
        columns = []
        for thruster in self.thrusters:
            force = thruster.direction_local * thruster.max_force_newtons
            # Torque = r x F from the thruster offset relative to the center of mass.
            torque = np.cross(thruster.position_local, force)
            columns.append(np.concatenate((force, torque)))
        return np.column_stack(columns)

    def set_thrust_vector(self, joy_input: Dict[str, float] | Iterable[float]) -> np.ndarray:
        """Convert normalized axis commands into per-thruster commands and PWM values."""
        if isinstance(joy_input, dict):
            desired_axes = np.array(
                [float(joy_input.get(axis, 0.0)) for axis in AXIS_ORDER], dtype=float
            )
        else:
            desired_axes = np.array(list(joy_input), dtype=float)
            if desired_axes.size != len(AXIS_ORDER):
                raise ValueError(f"Expected {len(AXIS_ORDER)} command axes, got {desired_axes.size}.")

        desired_axes = np.clip(desired_axes, -1.0, 1.0)
        # "Desired wrench" means the total 6-DOF force/torque request in the body frame.
        desired_wrench = desired_axes * self.axis_weights
        # The pseudo-inverse distributes that request across the available thrusters.
        self.command_vector = np.clip(self._mixer @ desired_wrench, -1.0, 1.0)
        self.pwm_vector = 1500.0 + 400.0 * self.command_vector
        return self.pwm_vector.copy()

    def thruster_force(self, thruster: Thruster, command: float) -> np.ndarray:
        return thruster.direction_local * thruster.max_force_newtons * command

    def calculate_thruster_power(
        self, local_linear_velocity: np.ndarray, local_angular_velocity: np.ndarray
    ) -> float:
        """Estimate mechanical power from force acting at moving thruster locations."""
        power = 0.0
        for thruster, command in zip(self.thrusters, self.command_vector):
            local_force = self.thruster_force(thruster, command)
            # Velocity at the thruster equals body translation plus rotational contribution.
            point_velocity = local_linear_velocity + np.cross(
                local_angular_velocity, thruster.position_local
            )
            power += float(np.dot(local_force, point_velocity))
        return power


class WaterPhysicsEngine:
    """Threaded Bullet stepping plus mathematical logging."""

    def __init__(
        self,
        client_id: int,
        body_id: int,
        controller: ROVController,
        params: ROVParameters,
        tank_bounds: TankBounds,
        csv_path: Path,
        step_hz: float = 240.0,
    ) -> None:
        self.client_id = client_id
        self.body_id = body_id
        self.controller = controller
        self.params = params
        self.tank_bounds = tank_bounds
        self.csv_path = csv_path
        self.time_step = 1.0 / step_hz
        self.state_lock = threading.Lock()
        self.bullet_lock = threading.Lock()
        self.stop_event = threading.Event()
        self.worker: Optional[threading.Thread] = None
        self.command_axes = {axis: 0.0 for axis in AXIS_ORDER}
        self.tuning = RuntimeTuning(
            rho=params.hydro.rho,
            drag_coefficient=params.hydro.drag_coefficient,
            buoyancy_multiplier=params.hydro.buoyancy_multiplier,
            sonar_max_range=6.0,
            num_lasers=5,
            detection_threshold=2.0,
            ping_frequency_hz=10.0,
            sonar_noise_level_m=0.04,
            zone_transition_smoothing_m=2.0,
        )
        self.start_time = 0.0
        self.prev_step_time = 0.0
        self.last_state: dict[str, object] = {}
        self.initial_reference_z = float(params.start_position[2] + params.geometry.com_offset_z)
        self.cumulative_thruster_work = 0.0
        self.cumulative_drag_loss = 0.0
        self.crosstrack_error_m = 0.0
        self.sonar_forward_confidence = 0.0
        self.sonar_mean_confidence = 0.0
        self.environment_rng = np.random.default_rng(31)
        self.latest_seabed_point = np.full(3, np.nan, dtype=float)
        self.latest_seabed_temperature_c = math.nan
        self.latest_seabed_luminosity_lux = math.nan
        self.latest_seabed_oxygen_pct = math.nan
        self.latest_seabed_ph = math.nan
        self.latest_seabed_sector_index = -1
        self.latest_seabed_sector_name = "none"
        self.geometry = params.geometry
        self.mass = params.geometry.mass
        self.inertia_diag = np.array(
            [
                params.geometry.inertia_xx,
                params.geometry.inertia_yy,
                params.geometry.inertia_zz,
            ],
            dtype=float,
        )

    def start(self) -> None:
        # The background thread keeps physics timing steady while the GUI stays responsive.
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_csv()
        self.start_time = time.perf_counter()
        self.prev_step_time = self.start_time
        self.worker = threading.Thread(target=self._run_loop, name="WaterPhysicsEngine", daemon=True)
        self.worker.start()

    def stop(self) -> None:
        self.stop_event.set()
        if self.worker is not None:
            self.worker.join(timeout=2.0)

    def update_command_axes(self, joy_input: Dict[str, float]) -> None:
        with self.state_lock:
            for axis in AXIS_ORDER:
                self.command_axes[axis] = float(joy_input.get(axis, 0.0))

    def update_runtime_tuning(self, tuning: RuntimeTuning) -> None:
        with self.state_lock:
            self.tuning = RuntimeTuning(
                rho=float(tuning.rho),
                drag_coefficient=float(tuning.drag_coefficient),
                buoyancy_multiplier=float(tuning.buoyancy_multiplier),
                sonar_max_range=float(tuning.sonar_max_range),
                num_lasers=int(tuning.num_lasers),
                detection_threshold=float(tuning.detection_threshold),
                ping_frequency_hz=float(tuning.ping_frequency_hz),
                sonar_noise_level_m=float(tuning.sonar_noise_level_m),
                zone_transition_smoothing_m=float(tuning.zone_transition_smoothing_m),
            )

    def update_path_metrics(self, crosstrack_error_m: float) -> None:
        with self.state_lock:
            self.crosstrack_error_m = float(crosstrack_error_m)

    def update_sonar_metrics(self, forward_confidence: float, mean_confidence: float) -> None:
        with self.state_lock:
            self.sonar_forward_confidence = float(forward_confidence)
            self.sonar_mean_confidence = float(mean_confidence)

    def update_seabed_mapping(
        self,
        seabed_point_world: Optional[np.ndarray],
        seabed_environment: Optional[dict[str, object]],
    ) -> None:
        with self.state_lock:
            if seabed_point_world is None or seabed_environment is None:
                self.latest_seabed_point = np.full(3, np.nan, dtype=float)
                self.latest_seabed_temperature_c = math.nan
                self.latest_seabed_luminosity_lux = math.nan
                self.latest_seabed_oxygen_pct = math.nan
                self.latest_seabed_ph = math.nan
                self.latest_seabed_sector_index = -1
                self.latest_seabed_sector_name = "none"
                return
            self.latest_seabed_point = np.array(seabed_point_world, dtype=float)
            self.latest_seabed_temperature_c = float(seabed_environment["temperature_c"])
            self.latest_seabed_luminosity_lux = float(seabed_environment["luminosity_lux"])
            self.latest_seabed_oxygen_pct = float(seabed_environment["oxygen_pct"])
            self.latest_seabed_ph = float(seabed_environment["ph"])
            self.latest_seabed_sector_index = int(seabed_environment["sector_index"])
            self.latest_seabed_sector_name = str(seabed_environment["sector_name"])

    def get_state_snapshot(self) -> dict[str, object]:
        # Return a copy so the main thread can inspect state safely during background stepping.
        with self.state_lock:
            snapshot = {}
            for key, value in self.last_state.items():
                snapshot[key] = value.copy() if isinstance(value, np.ndarray) else value
            return snapshot

    def _init_csv(self) -> None:
        # The CSV works like a black-box recorder for later validation and plotting.
        with self.csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(
                [
                    "time_s",
                    "x_m",
                    "y_m",
                    "z_m",
                    "roll_rad",
                    "pitch_rad",
                    "yaw_rad",
                    "u_mps",
                    "v_mps",
                    "w_mps",
                    "p_radps",
                    "q_radps",
                    "r_radps",
                    "kinetic_energy_j",
                    "potential_energy_j",
                    "total_energy_j",
                    "thruster_power_w",
                    "thruster_angular_power_w",
                    "drag_loss_w",
                    "cumulative_thruster_work_j",
                    "cumulative_drag_loss_j",
                    "energy_residual_j",
                    "restoring_moment_x_nm",
                    "restoring_moment_y_nm",
                    "restoring_moment_z_nm",
                    "rho_kgm3",
                    "drag_coefficient",
                    "buoyancy_multiplier",
                    "sonar_max_range_m",
                    "num_lasers",
                    "detection_threshold_m",
                    "ping_frequency_hz",
                    "sonar_noise_level_m",
                    "zone_transition_smoothing_m",
                    "sonar_forward_confidence",
                    "sonar_mean_confidence",
                    "sector_index",
                    "sector_name",
                    "temperature_c",
                    "luminosity_lux",
                    "oxygen_pct",
                    "ph",
                    "crosstrack_error_m",
                    "clearance_front_m",
                    "clearance_back_m",
                    "clearance_left_m",
                    "clearance_right_m",
                    "clearance_top_m",
                    "clearance_bottom_m",
                    "seabed_hit_x_m",
                    "seabed_hit_y_m",
                    "seabed_hit_z_m",
                    "seabed_sector_index",
                    "seabed_sector_name",
                    "seabed_temperature_c",
                    "seabed_luminosity_lux",
                    "seabed_oxygen_pct",
                    "seabed_ph",
                    "buoyancy_force_n",
                    "surge_cmd",
                    "sway_cmd",
                    "heave_cmd",
                    "roll_cmd",
                    "pitch_cmd",
                    "yaw_cmd",
                ]
            )

    def _run_loop(self) -> None:
        # Fixed-step integration makes the dynamics more repeatable across runs.
        next_tick = time.perf_counter()
        while not self.stop_event.is_set():
            loop_start = time.perf_counter()
            dt = max(loop_start - self.prev_step_time, self.time_step)
            with self.state_lock:
                joy_input = dict(self.command_axes)
                tuning = RuntimeTuning(
                    rho=self.tuning.rho,
                    drag_coefficient=self.tuning.drag_coefficient,
                    buoyancy_multiplier=self.tuning.buoyancy_multiplier,
                    sonar_max_range=self.tuning.sonar_max_range,
                    num_lasers=self.tuning.num_lasers,
                    detection_threshold=self.tuning.detection_threshold,
                    ping_frequency_hz=self.tuning.ping_frequency_hz,
                    sonar_noise_level_m=self.tuning.sonar_noise_level_m,
                    zone_transition_smoothing_m=self.tuning.zone_transition_smoothing_m,
                )

            self.controller.set_thrust_vector(joy_input)
            commands = np.array([joy_input[axis] for axis in AXIS_ORDER], dtype=float)

            with self.bullet_lock:
                # Apply forces first, then advance Bullet by one timestep.
                hydro_state = self.apply_hydrodynamics_locked(tuning)
                thrust_force, thrust_torque = self._apply_thrusters_locked()
                p.stepSimulation(physicsClientId=self.client_id)

            snapshot = self._build_snapshot(loop_start, hydro_state, thrust_force, thrust_torque, commands)
            self._append_csv(snapshot)

            with self.state_lock:
                self.last_state = snapshot

            self.prev_step_time = loop_start
            next_tick += self.time_step
            sleep_time = next_tick - time.perf_counter()
            if sleep_time > 0:
                time.sleep(sleep_time)
            else:
                next_tick = time.perf_counter()

    def _apply_thrusters_locked(self) -> tuple[np.ndarray, np.ndarray]:
        """Apply all current thruster forces in the body frame and sum the total wrench."""
        total_force = np.zeros(3, dtype=float)
        total_torque = np.zeros(3, dtype=float)
        for thruster, command in zip(self.controller.thrusters, self.controller.command_vector):
            local_force = self.controller.thruster_force(thruster, command)
            p.applyExternalForce(
                self.body_id,
                -1,
                local_force.tolist(),
                thruster.position_local.tolist(),
                p.LINK_FRAME,
                physicsClientId=self.client_id,
            )
            total_force += local_force
            total_torque += np.cross(thruster.position_local, local_force)
        return total_force, total_torque

    def apply_hydrodynamics_locked(self, tuning: RuntimeTuning) -> dict[str, np.ndarray | float]:
        """Compute buoyancy, translational drag, and rotational drag from the current state."""
        position, quaternion = p.getBasePositionAndOrientation(
            self.body_id, physicsClientId=self.client_id
        )
        linear_velocity_world, angular_velocity_world = p.getBaseVelocity(
            self.body_id, physicsClientId=self.client_id
        )
        rotation_world_from_body = np.array(
            p.getMatrixFromQuaternion(quaternion), dtype=float
        ).reshape(3, 3)
        linear_velocity_local = rotation_world_from_body.T @ np.array(linear_velocity_world, dtype=float)
        angular_velocity_local = rotation_world_from_body.T @ np.array(
            angular_velocity_world, dtype=float
        )

        # Buoyancy points upward in the world frame. We convert it into the body frame because
        # applyExternalForce below is using LINK_FRAME coordinates.
        buoyancy_magnitude = (
            tuning.rho * self.geometry.volume * GRAVITY * tuning.buoyancy_multiplier
        )
        buoyancy_world = np.array([0.0, 0.0, buoyancy_magnitude], dtype=float)
        buoyancy_local = rotation_world_from_body.T @ buoyancy_world

        projected_areas = self.geometry.projected_areas
        # Quadratic drag: F_d = -0.5 * rho * C_d * A * v * |v|
        drag_force_local = (
            -0.5
            * tuning.rho
            * tuning.drag_coefficient
            * projected_areas
            * linear_velocity_local
            * np.abs(linear_velocity_local)
        )
        angular_drag_local = (
            -0.5
            * tuning.rho
            * self.params.hydro.angular_drag_coefficient
            * self.geometry.angular_drag_area
            * angular_velocity_local
            * np.abs(angular_velocity_local)
        )

        # Applying buoyancy above the center of mass creates the restoring moment that helps
        # the ROV return upright after roll/pitch disturbances.
        p.applyExternalForce(
            self.body_id,
            -1,
            buoyancy_local.tolist(),
            self.params.hydro.center_of_buoyancy_local.tolist(),
            p.LINK_FRAME,
            physicsClientId=self.client_id,
        )
        p.applyExternalForce(
            self.body_id,
            -1,
            drag_force_local.tolist(),
            [0.0, 0.0, 0.0],
            p.LINK_FRAME,
            physicsClientId=self.client_id,
        )
        p.applyExternalTorque(
            self.body_id,
            -1,
            angular_drag_local.tolist(),
            p.LINK_FRAME,
            physicsClientId=self.client_id,
        )

        # This moment is a key teaching quantity when discussing static stability.
        restoring_moment_local = np.cross(self.params.hydro.center_of_buoyancy_local, buoyancy_local)
        # Drag loss is logged as positive dissipated power for easier energy accounting.
        drag_loss_power = -float(np.dot(drag_force_local, linear_velocity_local))
        drag_loss_power += -float(np.dot(angular_drag_local, angular_velocity_local))

        return {
            "position": np.array(position, dtype=float),
            "quaternion": np.array(quaternion, dtype=float),
            "linear_velocity_local": linear_velocity_local,
            "angular_velocity_local": angular_velocity_local,
            "buoyancy_force_local": buoyancy_local,
            "drag_force_local": drag_force_local,
            "angular_drag_local": angular_drag_local,
            "rho": tuning.rho,
            "drag_coefficient": tuning.drag_coefficient,
            "buoyancy_multiplier": tuning.buoyancy_multiplier,
            "buoyancy_magnitude": buoyancy_magnitude,
            "restoring_moment_local": restoring_moment_local,
            "drag_loss_power_w": max(drag_loss_power, 0.0),
        }

    def _build_snapshot(
        self,
        loop_time: float,
        hydro_state: dict[str, np.ndarray | float],
        thrust_force_local: np.ndarray,
        thrust_torque_local: np.ndarray,
        commands: np.ndarray,
    ) -> dict[str, object]:
        """Build one complete telemetry snapshot after each physics step."""
        position = hydro_state["position"]
        quaternion = hydro_state["quaternion"]
        linear_velocity_local = hydro_state["linear_velocity_local"]
        angular_velocity_local = hydro_state["angular_velocity_local"]
        euler = np.array(p.getEulerFromQuaternion(quaternion), dtype=float)

        # T = translational kinetic energy + rotational kinetic energy.
        kinetic_energy = 0.5 * self.mass * float(np.dot(linear_velocity_local, linear_velocity_local))
        kinetic_energy += 0.5 * float(
            np.dot(self.inertia_diag * angular_velocity_local, angular_velocity_local)
        )
        # V = mgh relative to the chosen depth reference at startup.
        potential_energy = self.mass * GRAVITY * (float(position[2]) - self.initial_reference_z)
        total_energy = kinetic_energy + potential_energy
        thruster_power = self.controller.calculate_thruster_power(
            linear_velocity_local, angular_velocity_local
        )
        thruster_angular_power = float(np.dot(thrust_torque_local, angular_velocity_local))
        drag_loss_power = float(hydro_state["drag_loss_power_w"])
        self.cumulative_thruster_work += thruster_power * self.time_step
        self.cumulative_drag_loss += drag_loss_power * self.time_step
        # Residual gives a quick sense of how far the simplified bookkeeping is from ideal closure.
        energy_residual = total_energy - self.cumulative_thruster_work + self.cumulative_drag_loss
        wall_clearances = self.tank_bounds.clearances(position)
        environmental_data = get_environmental_data(
            np.array(position, dtype=float),
            zone_transition_smoothing_m=self.tuning.zone_transition_smoothing_m,
        )

        return {
            "time_s": loop_time - self.start_time,
            "position": position,
            "quaternion": quaternion,
            "euler": euler,
            "linear_velocity_local": linear_velocity_local,
            "angular_velocity_local": angular_velocity_local,
            "kinetic_energy_j": kinetic_energy,
            "potential_energy_j": potential_energy,
            "total_energy_j": total_energy,
            "thruster_power_w": thruster_power,
            "thruster_angular_power_w": thruster_angular_power,
            "drag_loss_w": drag_loss_power,
            "cumulative_thruster_work_j": self.cumulative_thruster_work,
            "cumulative_drag_loss_j": self.cumulative_drag_loss,
            "energy_residual_j": energy_residual,
            "restoring_moment_local": hydro_state["restoring_moment_local"],
            "rho": hydro_state["rho"],
            "drag_coefficient": hydro_state["drag_coefficient"],
            "buoyancy_multiplier": hydro_state["buoyancy_multiplier"],
            "buoyancy_magnitude": hydro_state["buoyancy_magnitude"],
            "sonar_max_range": self.tuning.sonar_max_range,
            "num_lasers": self.tuning.num_lasers,
            "detection_threshold": self.tuning.detection_threshold,
            "ping_frequency_hz": self.tuning.ping_frequency_hz,
            "sonar_noise_level_m": self.tuning.sonar_noise_level_m,
            "zone_transition_smoothing_m": self.tuning.zone_transition_smoothing_m,
            "sonar_forward_confidence": self.sonar_forward_confidence,
            "sonar_mean_confidence": self.sonar_mean_confidence,
            "sector_index": environmental_data["sector_index"],
            "sector_name": environmental_data["sector_name"],
            "temperature_c": environmental_data["temperature_c"],
            "luminosity_lux": environmental_data["luminosity_lux"],
            "oxygen_pct": environmental_data["oxygen_pct"],
            "ph": environmental_data["ph"],
            "crosstrack_error_m": self.crosstrack_error_m,
            **wall_clearances,
            "seabed_point_world": self.latest_seabed_point.copy(),
            "seabed_sector_index": self.latest_seabed_sector_index,
            "seabed_sector_name": self.latest_seabed_sector_name,
            "seabed_temperature_c": self.latest_seabed_temperature_c,
            "seabed_luminosity_lux": self.latest_seabed_luminosity_lux,
            "seabed_oxygen_pct": self.latest_seabed_oxygen_pct,
            "seabed_ph": self.latest_seabed_ph,
            "commands": commands,
            "thrust_force_local": thrust_force_local,
            "thrust_torque_local": thrust_torque_local,
            "drag_force_local": hydro_state["drag_force_local"],
            "angular_drag_local": hydro_state["angular_drag_local"],
            "buoyancy_force_local": hydro_state["buoyancy_force_local"],
        }

    def _append_csv(self, snapshot: dict[str, object]) -> None:
        # Write one row per step so the run can be re-plotted or analyzed later.
        position = snapshot["position"]
        euler = snapshot["euler"]
        linear_velocity_local = snapshot["linear_velocity_local"]
        angular_velocity_local = snapshot["angular_velocity_local"]
        commands = snapshot["commands"]
        restoring_moment_local = snapshot["restoring_moment_local"]
        seabed_point_world = snapshot["seabed_point_world"]
        with self.csv_path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(
                [
                    f"{snapshot['time_s']:.5f}",
                    f"{position[0]:.5f}",
                    f"{position[1]:.5f}",
                    f"{position[2]:.5f}",
                    f"{euler[0]:.5f}",
                    f"{euler[1]:.5f}",
                    f"{euler[2]:.5f}",
                    *[f"{value:.5f}" for value in linear_velocity_local],
                    *[f"{value:.5f}" for value in angular_velocity_local],
                    f"{snapshot['kinetic_energy_j']:.5f}",
                    f"{snapshot['potential_energy_j']:.5f}",
                    f"{snapshot['total_energy_j']:.5f}",
                    f"{snapshot['thruster_power_w']:.5f}",
                    f"{snapshot['thruster_angular_power_w']:.5f}",
                    f"{snapshot['drag_loss_w']:.5f}",
                    f"{snapshot['cumulative_thruster_work_j']:.5f}",
                    f"{snapshot['cumulative_drag_loss_j']:.5f}",
                    f"{snapshot['energy_residual_j']:.5f}",
                    f"{restoring_moment_local[0]:.5f}",
                    f"{restoring_moment_local[1]:.5f}",
                    f"{restoring_moment_local[2]:.5f}",
                    f"{snapshot['rho']:.5f}",
                    f"{snapshot['drag_coefficient']:.5f}",
                    f"{snapshot['buoyancy_multiplier']:.5f}",
                    f"{snapshot['sonar_max_range']:.5f}",
                    f"{int(snapshot['num_lasers'])}",
                    f"{snapshot['detection_threshold']:.5f}",
                    f"{snapshot['ping_frequency_hz']:.5f}",
                    f"{snapshot['sonar_noise_level_m']:.5f}",
                    f"{snapshot['zone_transition_smoothing_m']:.5f}",
                    f"{snapshot['sonar_forward_confidence']:.5f}",
                    f"{snapshot['sonar_mean_confidence']:.5f}",
                    f"{int(snapshot['sector_index'])}",
                    str(snapshot["sector_name"]),
                    f"{snapshot['temperature_c']:.5f}",
                    f"{snapshot['luminosity_lux']:.5f}",
                    f"{snapshot['oxygen_pct']:.5f}",
                    f"{snapshot['ph']:.5f}",
                    f"{snapshot['crosstrack_error_m']:.5f}",
                    f"{snapshot['clearance_front_m']:.5f}",
                    f"{snapshot['clearance_back_m']:.5f}",
                    f"{snapshot['clearance_left_m']:.5f}",
                    f"{snapshot['clearance_right_m']:.5f}",
                    f"{snapshot['clearance_top_m']:.5f}",
                    f"{snapshot['clearance_bottom_m']:.5f}",
                    ""
                    if np.isnan(seabed_point_world[0])
                    else f"{float(seabed_point_world[0]):.5f}",
                    ""
                    if np.isnan(seabed_point_world[1])
                    else f"{float(seabed_point_world[1]):.5f}",
                    ""
                    if np.isnan(seabed_point_world[2])
                    else f"{float(seabed_point_world[2]):.5f}",
                    f"{int(snapshot['seabed_sector_index'])}",
                    str(snapshot["seabed_sector_name"]),
                    ""
                    if math.isnan(float(snapshot["seabed_temperature_c"]))
                    else f"{float(snapshot['seabed_temperature_c']):.5f}",
                    ""
                    if math.isnan(float(snapshot["seabed_luminosity_lux"]))
                    else f"{float(snapshot['seabed_luminosity_lux']):.5f}",
                    ""
                    if math.isnan(float(snapshot["seabed_oxygen_pct"]))
                    else f"{float(snapshot['seabed_oxygen_pct']):.5f}",
                    ""
                    if math.isnan(float(snapshot["seabed_ph"]))
                    else f"{float(snapshot['seabed_ph']):.5f}",
                    f"{snapshot['buoyancy_magnitude']:.5f}",
                    *[f"{value:.5f}" for value in commands],
                ]
            )


class ActiveSonar:
    """6-direction active-sonar model using conical multi-ray transducers."""

    def __init__(
        self,
        client_id: int,
        body_id: int,
        bullet_lock: threading.Lock,
        max_range: float = 6.0,
        seabed_z: float = -3.2,
    ) -> None:
        self.client_id = client_id
        self.body_id = body_id
        self.bullet_lock = bullet_lock
        self.sonar_max_range = max_range
        self.seabed_z = seabed_z
        self.sensor_origin_local = np.array([0.33, 0.0, 0.0], dtype=float)
        self.ray_count = 5
        self.noise_std_m = 0.04
        self.random = np.random.default_rng(17)
        self.debug_line_ids: list[int] = []
        self.filtered_direction_distances: dict[str, float] = {}
        self.cone_half_angle_rad = math.radians(10.0)

    def scan(
        self,
        snapshot: dict[str, object],
        draw_debug: bool,
        sonar_max_range: float,
        num_lasers: int,
        detection_threshold: float,
        sonar_noise_level_m: float,
    ) -> dict[str, object]:
        """Cast cone bundles in six directions and return the nearest reading in each cone."""
        self.sonar_max_range = float(np.clip(sonar_max_range, 5.0, 8.0))
        self.ray_count = int(np.clip(num_lasers, 5, 9))
        detection_threshold = float(np.clip(detection_threshold, 0.25, self.sonar_max_range))
        self.noise_std_m = float(np.clip(sonar_noise_level_m, 0.0, 0.5))

        rotation = np.array(p.getMatrixFromQuaternion(snapshot["quaternion"]), dtype=float).reshape(3, 3)
        position = np.array(snapshot["position"], dtype=float)
        origin_world = position + rotation @ self.sensor_origin_local

        # Each tuple defines the main look direction and two axes used to spread the cone.
        direction_specs = {
            "forward": (
                np.array([1.0, 0.0, 0.0], dtype=float),
                np.array([0.0, 1.0, 0.0], dtype=float),
                np.array([0.0, 0.0, 1.0], dtype=float),
            ),
            "backward": (
                np.array([-1.0, 0.0, 0.0], dtype=float),
                np.array([0.0, 1.0, 0.0], dtype=float),
                np.array([0.0, 0.0, 1.0], dtype=float),
            ),
            "left": (
                np.array([0.0, 1.0, 0.0], dtype=float),
                np.array([1.0, 0.0, 0.0], dtype=float),
                np.array([0.0, 0.0, 1.0], dtype=float),
            ),
            "right": (
                np.array([0.0, -1.0, 0.0], dtype=float),
                np.array([1.0, 0.0, 0.0], dtype=float),
                np.array([0.0, 0.0, 1.0], dtype=float),
            ),
            "up": (
                np.array([0.0, 0.0, 1.0], dtype=float),
                np.array([1.0, 0.0, 0.0], dtype=float),
                np.array([0.0, 1.0, 0.0], dtype=float),
            ),
            "down": (
                np.array([0.0, 0.0, -1.0], dtype=float),
                np.array([1.0, 0.0, 0.0], dtype=float),
                np.array([0.0, 1.0, 0.0], dtype=float),
            ),
        }

        def build_cone_offsets(ray_count: int) -> list[tuple[float, float, bool]]:
            # 5-9 samples across a cone cross-section. The outer samples define the visible beam spread.
            patterns = {
                5: [
                    (0.0, 0.0, False),
                    (1.0, 0.0, True),
                    (-1.0, 0.0, True),
                    (0.0, 1.0, True),
                    (0.0, -1.0, True),
                ],
                6: [
                    (0.0, 0.0, False),
                    (1.0, 0.0, True),
                    (-1.0, 0.0, True),
                    (0.0, 1.0, True),
                    (0.0, -1.0, True),
                    (1.0, 1.0, True),
                ],
                7: [
                    (0.0, 0.0, False),
                    (1.0, 0.0, True),
                    (-1.0, 0.0, True),
                    (0.0, 1.0, True),
                    (0.0, -1.0, True),
                    (1.0, 1.0, True),
                    (-1.0, -1.0, True),
                ],
                8: [
                    (0.0, 0.0, False),
                    (1.0, 0.0, True),
                    (-1.0, 0.0, True),
                    (0.0, 1.0, True),
                    (0.0, -1.0, True),
                    (1.0, 1.0, True),
                    (-1.0, -1.0, True),
                    (1.0, -1.0, True),
                ],
                9: [
                    (0.0, 0.0, False),
                    (1.0, 0.0, True),
                    (-1.0, 0.0, True),
                    (0.0, 1.0, True),
                    (0.0, -1.0, True),
                    (1.0, 1.0, True),
                    (-1.0, -1.0, True),
                    (1.0, -1.0, True),
                    (-1.0, 1.0, True),
                ],
            }
            return patterns[int(np.clip(ray_count, 5, 9))]

        cone_offsets = build_cone_offsets(self.ray_count)
        # The user asked for a 20 degree conical FOV, so we use a fixed 10 degree half-angle.
        cone_spread = math.tan(self.cone_half_angle_rad)
        ray_starts = []
        ray_ends = []
        ray_direction_labels: list[str] = []
        ray_outer_flags: list[bool] = []
        ray_directions_world: list[np.ndarray] = []
        for direction_name, (base_direction, axis_a, axis_b) in direction_specs.items():
            for offset_a, offset_b, is_outer in cone_offsets:
                # Offset the main beam direction slightly so we fill out a cone rather than a line.
                local_direction = (
                    base_direction
                    + cone_spread * offset_a * axis_a
                    + cone_spread * offset_b * axis_b
                )
                local_direction = local_direction / np.linalg.norm(local_direction)
                direction_world = rotation @ local_direction
                ray_starts.append(origin_world.tolist())
                ray_ends.append((origin_world + direction_world * self.sonar_max_range).tolist())
                ray_direction_labels.append(direction_name)
                ray_outer_flags.append(is_outer)
                ray_directions_world.append(direction_world)

        if len(self.debug_line_ids) != len(ray_starts):
            self.debug_line_ids = [-1] * len(ray_starts)

        with self.bullet_lock:
            results = p.rayTestBatch(ray_starts, ray_ends, physicsClientId=self.client_id)

        distances_by_direction: dict[str, list[float]] = {name: [] for name in direction_specs}
        confidences_by_direction: dict[str, list[float]] = {name: [] for name in direction_specs}
        seabed_hit_point_world: Optional[np.ndarray] = None
        seabed_hit_distance = math.inf
        hit_points = []
        all_distances = []
        for ray_end, result, direction_name, direction_world in zip(
            ray_ends, results, ray_direction_labels, ray_directions_world
        ):
            if result[0] >= 0:
                hit_point = np.array(result[3], dtype=float)
                geometric_distance = float(np.linalg.norm(hit_point - origin_world))
                hit_normal = np.array(result[4], dtype=float)
                hit_normal_norm = float(np.linalg.norm(hit_normal))
                if hit_normal_norm > 1e-9:
                    hit_normal /= hit_normal_norm
                incidence_alignment = float(
                    np.clip(np.dot(-direction_world, hit_normal), 0.0, 1.0)
                )
                sonar_confidence = float(incidence_alignment**1.5)
            else:
                hit_point = np.array(ray_end, dtype=float)
                geometric_distance = self.sonar_max_range
                sonar_confidence = 0.0
            # Active sonar measures travel time and converts it back into range using sound speed.
            round_trip_time_s = (2.0 * geometric_distance) / SPEED_OF_SOUND_WATER
            distance = 0.5 * SPEED_OF_SOUND_WATER * round_trip_time_s
            # Small noise makes the reading feel closer to real sonar than a perfect simulator.
            noisy_distance = float(
                np.clip(
                    distance + self.random.normal(0.0, self.noise_std_m),
                    0.0,
                    self.sonar_max_range,
                )
            )
            hit_points.append(hit_point)
            all_distances.append(noisy_distance)
            distances_by_direction[direction_name].append(noisy_distance)
            confidences_by_direction[direction_name].append(sonar_confidence)
            if (
                direction_name == "down"
                and result[0] >= 0
                and float(hit_point[2]) <= self.seabed_z + SEABED_HIT_TOLERANCE_M
                and geometric_distance < seabed_hit_distance
            ):
                seabed_hit_point_world = hit_point.copy()
                seabed_hit_distance = geometric_distance

        if draw_debug:
            with self.bullet_lock:
                for idx, (start, hit_point, distance, is_outer) in enumerate(
                    zip(ray_starts, hit_points, all_distances, ray_outer_flags)
                ):
                    # Red means an obstacle is inside the warning threshold, green means clear.
                    color = [1.0, 0.2, 0.2] if distance < detection_threshold else [0.15, 0.95, 0.35]
                    self.debug_line_ids[idx] = p.addUserDebugLine(
                        start,
                        hit_point.tolist(),
                        color,
                        lineWidth=2.6 if is_outer else 1.2,
                        lifeTime=0.18,
                        replaceItemUniqueId=self.debug_line_ids[idx],
                        physicsClientId=self.client_id,
                    )

        filtered_distances: dict[str, float] = {}
        direction_confidences: dict[str, float] = {}
        for direction_name, direction_distances in distances_by_direction.items():
            # We use the minimum distance in each cone because the nearest obstacle is the risky one.
            minimum_distance = float(min(direction_distances))
            previous_distance = self.filtered_direction_distances.get(direction_name, minimum_distance)
            # React faster when something gets closer, relax more slowly when it moves away.
            smoothing_alpha = 0.78 if minimum_distance < previous_distance else 0.35
            filtered_distance = float(
                np.clip(
                    smoothing_alpha * minimum_distance
                    + (1.0 - smoothing_alpha) * previous_distance,
                    0.0,
                    self.sonar_max_range,
                )
            )
            self.filtered_direction_distances[direction_name] = filtered_distance
            filtered_distances[direction_name] = filtered_distance
            shortest_index = int(np.argmin(direction_distances))
            direction_confidences[direction_name] = float(
                confidences_by_direction[direction_name][shortest_index]
            )

        return {
            "distances_m": np.array(all_distances, dtype=float),
            "forward_m": filtered_distances["forward"],
            "backward_m": filtered_distances["backward"],
            "left_m": filtered_distances["left"],
            "right_m": filtered_distances["right"],
            "up_m": filtered_distances["up"],
            "down_m": filtered_distances["down"],
            "forward_confidence": direction_confidences["forward"],
            "backward_confidence": direction_confidences["backward"],
            "left_confidence": direction_confidences["left"],
            "right_confidence": direction_confidences["right"],
            "up_confidence": direction_confidences["up"],
            "down_confidence": direction_confidences["down"],
            "mean_confidence": float(np.mean(list(direction_confidences.values()))),
            "num_lasers": self.ray_count,
            "sonar_max_range": self.sonar_max_range,
            "detection_threshold": detection_threshold,
            "seabed_point_world": None if seabed_hit_point_world is None else seabed_hit_point_world,
        }


class SeabedMapLogger:
    """Records bathymetric sonar hits and leaves a persistent point cloud in the GUI."""

    def __init__(
        self,
        client_id: int,
        bullet_lock: threading.Lock,
        csv_path: Path,
        cell_size_m: float = SEABED_CELL_SIZE_M,
    ) -> None:
        self.client_id = client_id
        self.bullet_lock = bullet_lock
        self.csv_path = csv_path
        self.cell_size_m = cell_size_m
        self.total_points_captured = 0
        self.unique_cells: set[tuple[int, int]] = set()

    def start(self) -> None:
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        with self.csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(
                [
                    "time_s",
                    "x_m",
                    "y_m",
                    "z_m",
                    "sector_index",
                    "sector_name",
                    "temperature_c",
                    "luminosity_lux",
                    "oxygen_pct",
                    "ph",
                ]
            )

    def _cell_key(self, point_world: np.ndarray) -> tuple[int, int]:
        return (
            int(math.floor(float(point_world[0]) / self.cell_size_m)),
            int(math.floor(float(point_world[1]) / self.cell_size_m)),
        )

    def record_hit(
        self,
        time_s: float,
        point_world: np.ndarray,
        environmental_data: dict[str, object],
        draw_debug: bool,
    ) -> None:
        point_world = np.array(point_world, dtype=float)
        self.total_points_captured += 1
        cell_key = self._cell_key(point_world)
        is_new_cell = cell_key not in self.unique_cells
        self.unique_cells.add(cell_key)

        with self.csv_path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(
                [
                    f"{time_s:.5f}",
                    f"{point_world[0]:.5f}",
                    f"{point_world[1]:.5f}",
                    f"{point_world[2]:.5f}",
                    int(environmental_data["sector_index"]),
                    str(environmental_data["sector_name"]),
                    f"{float(environmental_data['temperature_c']):.5f}",
                    f"{float(environmental_data['luminosity_lux']):.5f}",
                    f"{float(environmental_data['oxygen_pct']):.5f}",
                    f"{float(environmental_data['ph']):.5f}",
                ]
            )

        if draw_debug and is_new_cell:
            sector_index = int(environmental_data["sector_index"])
            point_color = {
                1: [0.95, 0.88, 0.35],
                2: [0.35, 0.90, 0.95],
                3: [0.30, 0.55, 0.95],
                4: [0.85, 0.30, 0.95],
            }.get(sector_index, [0.90, 0.90, 0.90])
            with self.bullet_lock:
                p.addUserDebugPoints(
                    [point_world.tolist()],
                    [point_color],
                    pointSize=6.0,
                    lifeTime=0.0,
                    physicsClientId=self.client_id,
                )


class SimulationDashboard:
    """Live side-by-side validation plots."""

    def __init__(self, enabled: bool) -> None:
        self.enabled = enabled
        self.times = deque(maxlen=1200)
        self.u_values = deque(maxlen=1200)
        self.v_values = deque(maxlen=1200)
        self.w_values = deque(maxlen=1200)
        self.kinetic_values = deque(maxlen=1200)
        self.potential_values = deque(maxlen=1200)
        self.total_values = deque(maxlen=1200)
        self.last_refresh = 0.0
        self.refresh_period = 0.25
        self.figure: Optional[plt.Figure] = None
        self.velocity_axis = None
        self.energy_axis = None
        self.velocity_lines = ()
        self.energy_lines = ()

    def start(self) -> None:
        if not self.enabled:
            return
        plt.ion()
        # Two subplots mirror the typical deliverables: response plot and energy plot.
        self.figure, (self.velocity_axis, self.energy_axis) = plt.subplots(1, 2, figsize=(13, 5))
        self.figure.canvas.manager.set_window_title("ROV Mathematical Validation Dashboard")

        u_line, = self.velocity_axis.plot([], [], label="u", color="#0077b6")
        v_line, = self.velocity_axis.plot([], [], label="v", color="#ff7b00")
        w_line, = self.velocity_axis.plot([], [], label="w", color="#43aa8b")
        self.velocity_lines = (u_line, v_line, w_line)
        self.velocity_axis.set_title("Plot A: Linear Velocity vs Time")
        self.velocity_axis.set_xlabel("Time (s)")
        self.velocity_axis.set_ylabel("Velocity (m/s)")
        self.velocity_axis.grid(True, alpha=0.3)
        self.velocity_axis.legend(loc="upper right")

        kinetic_line, = self.energy_axis.plot([], [], label="T", color="#2a9d8f")
        potential_line, = self.energy_axis.plot([], [], label="V", color="#8d99ae")
        total_line, = self.energy_axis.plot([], [], label="E", color="#d62828")
        self.energy_lines = (kinetic_line, potential_line, total_line)
        self.energy_axis.set_title("Plot B: Energy vs Time")
        self.energy_axis.set_xlabel("Time (s)")
        self.energy_axis.set_ylabel("Energy (J)")
        self.energy_axis.grid(True, alpha=0.3)
        self.energy_axis.legend(loc="upper right")

        self.figure.tight_layout()
        self.figure.show()

    def push_snapshot(self, snapshot: dict[str, object]) -> None:
        # Store only the values needed for plotting to keep updates lightweight.
        linear_velocity_local = snapshot["linear_velocity_local"]
        self.times.append(float(snapshot["time_s"]))
        self.u_values.append(float(linear_velocity_local[0]))
        self.v_values.append(float(linear_velocity_local[1]))
        self.w_values.append(float(linear_velocity_local[2]))
        self.kinetic_values.append(float(snapshot["kinetic_energy_j"]))
        self.potential_values.append(float(snapshot["potential_energy_j"]))
        self.total_values.append(float(snapshot["total_energy_j"]))

    def maybe_refresh(self, force: bool = False) -> None:
        # Refresh periodically instead of every frame so plotting does not lag the simulation.
        if not self.enabled or self.figure is None or not self.times:
            return
        now = time.perf_counter()
        if not force and now - self.last_refresh < self.refresh_period:
            return
        self.last_refresh = now

        times = list(self.times)
        self.velocity_lines[0].set_data(times, list(self.u_values))
        self.velocity_lines[1].set_data(times, list(self.v_values))
        self.velocity_lines[2].set_data(times, list(self.w_values))
        self.energy_lines[0].set_data(times, list(self.kinetic_values))
        self.energy_lines[1].set_data(times, list(self.potential_values))
        self.energy_lines[2].set_data(times, list(self.total_values))

        for axis in (self.velocity_axis, self.energy_axis):
            axis.relim()
            axis.autoscale_view()

        self.figure.canvas.draw_idle()
        self.figure.canvas.flush_events()
        plt.pause(0.001)

    def save(self, output_path: Path) -> None:
        # Save a static image at the end so headless runs still produce a dashboard artifact.
        if not self.times:
            return
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if self.figure is not None:
            self.maybe_refresh(force=True)
            self.figure.savefig(output_path, dpi=160)
            return

        fig, (velocity_axis, energy_axis) = plt.subplots(1, 2, figsize=(13, 5))
        velocity_axis.plot(self.times, self.u_values, label="u", color="#0077b6")
        velocity_axis.plot(self.times, self.v_values, label="v", color="#ff7b00")
        velocity_axis.plot(self.times, self.w_values, label="w", color="#43aa8b")
        velocity_axis.set_title("Plot A: Linear Velocity vs Time")
        velocity_axis.set_xlabel("Time (s)")
        velocity_axis.set_ylabel("Velocity (m/s)")
        velocity_axis.grid(True, alpha=0.3)
        velocity_axis.legend(loc="upper right")

        energy_axis.plot(self.times, self.kinetic_values, label="T", color="#2a9d8f")
        energy_axis.plot(self.times, self.potential_values, label="V", color="#8d99ae")
        energy_axis.plot(self.times, self.total_values, label="E", color="#d62828")
        energy_axis.set_title("Plot B: Energy vs Time")
        energy_axis.set_xlabel("Time (s)")
        energy_axis.set_ylabel("Energy (J)")
        energy_axis.grid(True, alpha=0.3)
        energy_axis.legend(loc="upper right")
        fig.tight_layout()
        fig.savefig(output_path, dpi=160)
        plt.close(fig)

    def close(self) -> None:
        if self.figure is not None:
            plt.close(self.figure)


def world_setup(client_id: int) -> tuple[TankBounds, set[int]]:
    """Create the enclosed water tank and return its geometric limits."""
    p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=client_id)
    p.resetSimulation(physicsClientId=client_id)
    p.setGravity(0.0, 0.0, -GRAVITY, physicsClientId=client_id)
    p.setTimeStep(1.0 / 240.0, physicsClientId=client_id)
    p.setPhysicsEngineParameter(
        fixedTimeStep=1.0 / 240.0,
        numSolverIterations=120,
        numSubSteps=1,
        physicsClientId=client_id,
    )

    tank_bounds = TankBounds(
        x_min=-31.5,
        x_max=31.5,
        y_min=-3.0,
        y_max=3.0,
        z_min=-3.2,
        z_max=0.2,
    )
    wall_ids: set[int] = set()

    # A thin translucent box at the top acts as the visible water surface.
    water_visual = p.createVisualShape(
        p.GEOM_BOX,
        halfExtents=[
            (tank_bounds.x_max - tank_bounds.x_min) / 2.0,
            (tank_bounds.y_max - tank_bounds.y_min) / 2.0,
            0.01,
        ],
        rgbaColor=[0.12, 0.45, 0.92, 0.16],
        physicsClientId=client_id,
    )
    p.createMultiBody(
        baseMass=0.0,
        baseCollisionShapeIndex=-1,
        baseVisualShapeIndex=water_visual,
        basePosition=[0.0, 0.0, tank_bounds.z_max],
        physicsClientId=client_id,
    )

    for grid_x in np.linspace(tank_bounds.x_min + 1.5, tank_bounds.x_max - 1.5, 11):
        # The floor grid helps viewers judge motion and distance in the 3D scene.
        p.addUserDebugLine(
            [grid_x, tank_bounds.y_min, tank_bounds.z_min],
            [grid_x, tank_bounds.y_max, tank_bounds.z_min],
            [0.64, 0.64, 0.64],
            0.55,
            physicsClientId=client_id,
        )
    for grid_y in np.linspace(tank_bounds.y_min, tank_bounds.y_max, 7):
        p.addUserDebugLine(
            [tank_bounds.x_min, grid_y, tank_bounds.z_min],
            [tank_bounds.x_max, grid_y, tank_bounds.z_min],
            [0.64, 0.64, 0.64],
            0.55,
            physicsClientId=client_id,
        )

    blue = [0.18, 0.56, 0.96, 0.16]
    half_x = (tank_bounds.x_max - tank_bounds.x_min) / 2.0
    half_y = (tank_bounds.y_max - tank_bounds.y_min) / 2.0
    half_z = (tank_bounds.z_max - tank_bounds.z_min) / 2.0
    # Six panels form the test tank: left, right, back, front, top, and bottom.
    panels = [
        ([0.0, tank_bounds.y_min, (tank_bounds.z_min + tank_bounds.z_max) / 2.0], [half_x, 0.05, half_z]),
        ([0.0, tank_bounds.y_max, (tank_bounds.z_min + tank_bounds.z_max) / 2.0], [half_x, 0.05, half_z]),
        ([tank_bounds.x_min, 0.0, (tank_bounds.z_min + tank_bounds.z_max) / 2.0], [0.05, half_y, half_z]),
        ([tank_bounds.x_max, 0.0, (tank_bounds.z_min + tank_bounds.z_max) / 2.0], [0.05, half_y, half_z]),
        ([0.0, 0.0, tank_bounds.z_max], [half_x, half_y, 0.05]),
        ([0.0, 0.0, tank_bounds.z_min], [half_x, half_y, 0.05]),
    ]
    for position, half_extents in panels:
        collision = p.createCollisionShape(
            p.GEOM_BOX, halfExtents=half_extents, physicsClientId=client_id
        )
        visual = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=half_extents,
            rgbaColor=blue,
            physicsClientId=client_id,
        )
        wall_id = p.createMultiBody(
            baseMass=0.0,
            baseCollisionShapeIndex=collision,
            baseVisualShapeIndex=visual,
            basePosition=position,
            physicsClientId=client_id,
        )
        wall_ids.add(wall_id)

    zone_y_half = (tank_bounds.y_max - tank_bounds.y_min) / 2.0
    zone_z_half = (tank_bounds.z_max - tank_bounds.z_min) / 2.0
    for sector_index, sector_name, x_min, x_max, tint_rgba in SECTOR_DEFINITIONS:
        clamped_x_min = max(x_min, tank_bounds.x_min)
        clamped_x_max = min(x_max, tank_bounds.x_max)
        if clamped_x_max <= clamped_x_min:
            continue
        zone_half_x = 0.5 * (clamped_x_max - clamped_x_min)
        zone_center_x = 0.5 * (clamped_x_min + clamped_x_max)
        zone_visual = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=[zone_half_x, zone_y_half, zone_z_half],
            rgbaColor=tint_rgba,
            physicsClientId=client_id,
        )
        p.createMultiBody(
            baseMass=0.0,
            baseCollisionShapeIndex=-1,
            baseVisualShapeIndex=zone_visual,
            basePosition=[zone_center_x, 0.0, 0.5 * (tank_bounds.z_min + tank_bounds.z_max)],
            physicsClientId=client_id,
        )
        p.addUserDebugText(
            f"Zone {sector_index}: {sector_name}",
            [zone_center_x, tank_bounds.y_min + 0.18, tank_bounds.z_max - 0.20],
            textColorRGB=tint_rgba[:3],
            textSize=1.0,
            physicsClientId=client_id,
        )

    return tank_bounds, wall_ids


def generate_underwater_course(
    client_id: int, gate_count: int = 10, area_size: float = 60.0, seed: int = 11
) -> tuple[list[int], list[GateTarget]]:
    """Build the gate mission plus obstacles used by the navigation tests."""
    rng = np.random.default_rng(seed)
    obstacle_ids = []
    gate_targets: list[GateTarget] = []
    half_area = area_size / 2.0

    gate_width = 1.80
    gate_height = 1.50
    frame_thickness = 0.07
    ring_spacing = 6.0
    course_start_x = -27.0
    x_positions = np.array(
        [course_start_x + ring_spacing * gate_index for gate_index in range(gate_count)], dtype=float
    )

    for gate_index, x in enumerate(x_positions):
        # Gates vary a little in height and lateral offset so the path is interesting but structured.
        y = float(0.55 * math.sin(0.45 * gate_index))
        z = float(-1.35 + 0.22 * math.cos(0.40 * gate_index))
        yaw = float(rng.uniform(-0.22, 0.22))
        rotation = np.array(
            [
                [math.cos(yaw), -math.sin(yaw), 0.0],
                [math.sin(yaw), math.cos(yaw), 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=float,
        )
        gate_center = np.array([x, y, z], dtype=float)
        gate_targets.append(
            GateTarget(
                center_world=gate_center.copy(),
                yaw=yaw,
                width=gate_width,
                height=gate_height,
            )
        )
        segments = [
            (np.array([0.0, gate_width / 2.0, 0.0]), [frame_thickness, frame_thickness, gate_height / 2.0]),
            (np.array([0.0, -gate_width / 2.0, 0.0]), [frame_thickness, frame_thickness, gate_height / 2.0]),
            (np.array([0.0, 0.0, gate_height / 2.0]), [frame_thickness, gate_width / 2.0, frame_thickness]),
            (np.array([0.0, 0.0, -gate_height / 2.0]), [frame_thickness, gate_width / 2.0, frame_thickness]),
        ]

        for local_offset, half_extents in segments:
            collision = p.createCollisionShape(
                p.GEOM_BOX, halfExtents=half_extents, physicsClientId=client_id
            )
            visual = p.createVisualShape(
                p.GEOM_BOX,
                halfExtents=half_extents,
                rgbaColor=[0.82, 0.74, 0.18, 0.96],
                physicsClientId=client_id,
            )
            segment_world = gate_center + rotation @ local_offset
            obstacle_id = p.createMultiBody(
                baseMass=0.0,
                baseCollisionShapeIndex=collision,
                baseVisualShapeIndex=visual,
                basePosition=segment_world.tolist(),
                baseOrientation=p.getQuaternionFromEuler([0.0, 0.0, yaw]),
                physicsClientId=client_id,
            )
            obstacle_ids.append(obstacle_id)

        if gate_index < gate_count - 1:
            next_x = x_positions[gate_index + 1]
            mid_x = float(0.5 * (x + next_x))

            # A vertical cylinder forces lateral avoidance or yaw adjustment.
            cylinder_y = float(np.clip(y + rng.choice([-1.0, 1.0]), -1.6, 1.6))
            cylinder_collision = p.createCollisionShape(
                p.GEOM_CYLINDER, radius=0.18, height=1.6, physicsClientId=client_id
            )
            cylinder_visual = p.createVisualShape(
                p.GEOM_CYLINDER,
                radius=0.18,
                length=1.6,
                rgbaColor=[0.86, 0.46, 0.16, 0.90],
                physicsClientId=client_id,
            )
            obstacle_ids.append(
                p.createMultiBody(
                    baseMass=0.0,
                    baseCollisionShapeIndex=cylinder_collision,
                    baseVisualShapeIndex=cylinder_visual,
                    basePosition=[mid_x, cylinder_y, -1.2],
                    baseOrientation=p.getQuaternionFromEuler([math.pi / 2.0, 0.0, 0.0]),
                    physicsClientId=client_id,
                )
            )

            # A horizontal bar forces the vehicle to move above or below it.
            bar_z = float(np.clip(z + rng.choice([-0.40, 0.40]), -2.0, -0.60))
            bar_collision = p.createCollisionShape(
                p.GEOM_BOX, halfExtents=[0.10, 0.95, 0.08], physicsClientId=client_id
            )
            bar_visual = p.createVisualShape(
                p.GEOM_BOX,
                halfExtents=[0.10, 0.95, 0.08],
                rgbaColor=[0.72, 0.30, 0.24, 0.92],
                physicsClientId=client_id,
            )
            obstacle_ids.append(
                p.createMultiBody(
                    baseMass=0.0,
                    baseCollisionShapeIndex=bar_collision,
                    baseVisualShapeIndex=bar_visual,
                    basePosition=[mid_x, 0.0, bar_z],
                    physicsClientId=client_id,
                )
            )

            # Small debris adds realism and forces fine corrections without fully blocking the route.
            for _ in range(2):
                debris_shape = p.GEOM_BOX if rng.random() < 0.5 else p.GEOM_SPHERE
                debris_x = float(rng.uniform(x + 1.0, next_x - 1.0))
                debris_y = float(rng.uniform(-1.5, 1.5))
                debris_z = float(rng.uniform(-2.0, -0.5))
                if debris_shape == p.GEOM_BOX:
                    extents = rng.uniform(0.08, 0.14, size=3)
                    collision = p.createCollisionShape(
                        p.GEOM_BOX, halfExtents=extents.tolist(), physicsClientId=client_id
                    )
                    visual = p.createVisualShape(
                        p.GEOM_BOX,
                        halfExtents=extents.tolist(),
                        rgbaColor=[0.42, 0.70, 0.68, 0.88],
                        physicsClientId=client_id,
                    )
                else:
                    radius = float(rng.uniform(0.08, 0.13))
                    collision = p.createCollisionShape(
                        p.GEOM_SPHERE, radius=radius, physicsClientId=client_id
                    )
                    visual = p.createVisualShape(
                        p.GEOM_SPHERE,
                        radius=radius,
                        rgbaColor=[0.52, 0.80, 0.62, 0.88],
                        physicsClientId=client_id,
                    )
                obstacle_ids.append(
                    p.createMultiBody(
                        baseMass=0.0,
                        baseCollisionShapeIndex=collision,
                        baseVisualShapeIndex=visual,
                        basePosition=[debris_x, debris_y, debris_z],
                        physicsClientId=client_id,
                    )
                )

    if gate_targets:
        # Start and goal markers make the mission easier to read in the GUI.
        start_visual = p.createVisualShape(
            p.GEOM_SPHERE,
            radius=0.40,
            rgbaColor=[0.10, 0.85, 0.55, 0.20],
            physicsClientId=client_id,
        )
        goal_visual = p.createVisualShape(
            p.GEOM_SPHERE,
            radius=0.50,
            rgbaColor=[0.98, 0.36, 0.24, 0.24],
            physicsClientId=client_id,
        )
        p.createMultiBody(
            baseMass=0.0,
            baseCollisionShapeIndex=-1,
            baseVisualShapeIndex=start_visual,
            basePosition=gate_targets[0].center_world.tolist(),
            physicsClientId=client_id,
        )
        p.createMultiBody(
            baseMass=0.0,
            baseCollisionShapeIndex=-1,
            baseVisualShapeIndex=goal_visual,
            basePosition=gate_targets[-1].center_world.tolist(),
            physicsClientId=client_id,
        )

    return obstacle_ids, gate_targets


def create_rov(client_id: int, params: ROVParameters, urdf_path: Path) -> int:
    """Load the URDF model and apply low dry friction so water forces dominate behavior."""
    start_quaternion = p.getQuaternionFromEuler(params.start_rpy.tolist())
    body_id = p.loadURDF(
        str(urdf_path),
        basePosition=params.start_position.tolist(),
        baseOrientation=start_quaternion,
        flags=p.URDF_USE_INERTIA_FROM_FILE,
        physicsClientId=client_id,
    )
    p.changeDynamics(
        body_id,
        -1,
        linearDamping=0.0,
        angularDamping=0.0,
        lateralFriction=0.25,
        spinningFriction=0.01,
        rollingFriction=0.01,
        physicsClientId=client_id,
    )
    return body_id


def key_is_down(events: dict[int, int], *codes: int) -> bool:
    """True while a key is being held."""
    return any(events.get(code, 0) & p.KEY_IS_DOWN for code in codes)


def key_triggered(events: dict[int, int], *codes: int) -> bool:
    """True only on the frame where a key was first pressed."""
    return any(events.get(code, 0) & p.KEY_WAS_TRIGGERED for code in codes)


def handle_mode_switch(current_mode: str, events: dict[int, int]) -> str:
    """Switch between automatic and manual mode using the number keys."""
    if key_triggered(events, ord("1")):
        return AUTOMATIC_MODE
    if key_triggered(events, ord("2")):
        return MANUAL_MODE
    return current_mode


def manual_keyboard_command(events: dict[int, int]) -> dict[str, float]:
    """Map keyboard input to normalized body-axis commands."""
    surge = 0.0
    sway = 0.0
    heave = 0.0
    roll = 0.0

    if key_is_down(events, ord("w"), ord("W")):
        surge += 0.72
    if key_is_down(events, ord("s"), ord("S")):
        surge -= 0.55
    if key_is_down(events, ord("a"), ord("A")):
        sway += 0.60
    if key_is_down(events, ord("d"), ord("D")):
        sway -= 0.60
    if key_is_down(events, p.B3G_SHIFT):
        heave += 0.55
    if key_is_down(events, p.B3G_CONTROL):
        heave -= 0.55
    if key_is_down(events, ord("q"), ord("Q")):
        roll += 0.55
    if key_is_down(events, ord("e"), ord("E")):
        roll -= 0.55

    return {
        "surge": surge,
        "sway": sway,
        "heave": heave,
        "roll": roll,
        "pitch": 0.0,
        "yaw": 0.0,
    }


def select_next_gate(
    position: np.ndarray, gate_targets: list[GateTarget], current_gate_index: int
) -> tuple[Optional[GateTarget], int]:
    """Pick the next gate ahead of the ROV based on the current x-position."""
    if not gate_targets:
        return None, 0

    gate_index = int(np.clip(current_gate_index, 0, len(gate_targets) - 1))
    while gate_index < len(gate_targets) - 1 and position[0] > gate_targets[gate_index].center_world[0] + 0.45:
        gate_index += 1
    return gate_targets[gate_index], gate_index


class MissionManager:
    """Tracks mission progression, impacts, and completion metrics."""

    def __init__(
        self,
        gate_targets: list[GateTarget],
        summary_path: Path,
        impact_log_path: Path,
        wall_ids: set[int],
        start_position: np.ndarray,
        tank_bounds: TankBounds,
        seabed_cell_size_m: float = SEABED_CELL_SIZE_M,
    ) -> None:
        self.gate_targets = gate_targets
        self.summary_path = summary_path
        self.impact_log_path = impact_log_path
        self.wall_ids = wall_ids
        self.tank_bounds = tank_bounds
        self.seabed_cell_size_m = seabed_cell_size_m
        self.waypoints = [np.array(start_position, dtype=float)] + [
            gate.center_world.copy() for gate in gate_targets
        ]
        self.status = MissionStatus()
        self._last_gate_local_x: Optional[float] = None
        self._last_time_s: Optional[float] = None
        self._active_contact_ids: set[int] = set()
        self.total_points_captured = 0
        self.unique_seabed_cells: set[tuple[int, int]] = set()

    def current_gate_target(self) -> Optional[GateTarget]:
        # Once the last gate is cleared, there is no longer an active gate target.
        if self.status.current_gate_index >= len(self.gate_targets):
            return None
        return self.gate_targets[self.status.current_gate_index]

    def current_path_guidance(self, position: np.ndarray) -> PathGuidance:
        """Compute the active line segment, next waypoint, and crosstrack error."""
        if len(self.waypoints) == 1:
            waypoint = self.waypoints[0].copy()
            zero = np.zeros(3, dtype=float)
            return PathGuidance(0, waypoint, waypoint, waypoint, zero, zero, 0.0)

        active_waypoint_index = min(self.status.current_gate_index + 1, len(self.waypoints) - 1)
        segment_start = self.waypoints[max(active_waypoint_index - 1, 0)]
        segment_end = self.waypoints[active_waypoint_index]
        segment_vector = segment_end - segment_start
        segment_norm_sq = float(np.dot(segment_vector, segment_vector))
        if segment_norm_sq <= 1e-9:
            projection = segment_start.copy()
        else:
            # Project the current position onto the active path segment.
            projection_scale = float(
                np.clip(np.dot(position - segment_start, segment_vector) / segment_norm_sq, 0.0, 1.0)
            )
            projection = segment_start + projection_scale * segment_vector

        crosstrack_vector = projection - position
        heading_vector = segment_end - position
        return PathGuidance(
            active_waypoint_index=active_waypoint_index,
            active_waypoint_world=segment_end.copy(),
            segment_start_world=segment_start.copy(),
            segment_end_world=segment_end.copy(),
            heading_vector_world=heading_vector,
            crosstrack_vector_world=crosstrack_vector,
            crosstrack_error_m=float(np.linalg.norm(crosstrack_vector)),
        )

    def update(self, snapshot: dict[str, object], contact_points: list[tuple]) -> None:
        """Advance mission state using pose, energy, and contact information."""
        time_s = float(snapshot["time_s"])
        position = np.array(snapshot["position"], dtype=float)

        if self.status.start_time_s is None and self.gate_targets:
            start_gate = self.gate_targets[0]
            if float(np.linalg.norm(position - start_gate.center_world)) <= self.status.start_radius_m:
                self.status.start_time_s = time_s

        self._check_gate_pass(snapshot)
        self._check_completion(snapshot)
        self._update_energy(snapshot)
        self._register_impacts(contact_points, time_s)

    def register_seabed_hit(self, point_world: np.ndarray) -> None:
        point_world = np.array(point_world, dtype=float)
        self.total_points_captured += 1
        cell_key = (
            int(math.floor(float(point_world[0]) / self.seabed_cell_size_m)),
            int(math.floor(float(point_world[1]) / self.seabed_cell_size_m)),
        )
        self.unique_seabed_cells.add(cell_key)

    def scanning_coverage_pct(self) -> float:
        total_x_cells = max(
            int(math.ceil((self.tank_bounds.x_max - self.tank_bounds.x_min) / self.seabed_cell_size_m)),
            1,
        )
        total_y_cells = max(
            int(math.ceil((self.tank_bounds.y_max - self.tank_bounds.y_min) / self.seabed_cell_size_m)),
            1,
        )
        total_cells = total_x_cells * total_y_cells
        return 100.0 * len(self.unique_seabed_cells) / total_cells

    def mapping_efficiency(self) -> float:
        return self.total_points_captured / max(self.status.total_energy_consumed_j, 1e-6)

    def _check_gate_pass(self, snapshot: dict[str, object]) -> None:
        # Transform the ROV position into the local frame of the current gate.
        gate_target = self.current_gate_target()
        if gate_target is None:
            return

        position = np.array(snapshot["position"], dtype=float)
        yaw = gate_target.yaw
        rotation_gate_from_world = np.array(
            [
                [math.cos(yaw), math.sin(yaw), 0.0],
                [-math.sin(yaw), math.cos(yaw), 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=float,
        )
        local_position = rotation_gate_from_world @ (position - gate_target.center_world)
        gate_margin_y = gate_target.width / 2.0 - 0.30
        gate_margin_z = gate_target.height / 2.0 - 0.30
        within_gate = abs(local_position[1]) <= gate_margin_y and abs(local_position[2]) <= gate_margin_z

        if self._last_gate_local_x is None:
            self._last_gate_local_x = float(local_position[0])
            return

        crossed_plane = self._last_gate_local_x < 0.0 <= float(local_position[0])
        if within_gate and crossed_plane:
            # A gate is counted only if the ROV crosses the gate plane while staying inside the opening.
            gate_time = float(snapshot["time_s"])
            self.status.gate_pass_times.append(gate_time)
            self.status.current_gate_index += 1
            if self.status.start_time_s is None:
                self.status.start_time_s = gate_time
            self._last_gate_local_x = None
            return

        self._last_gate_local_x = float(local_position[0])

    def _check_completion(self, snapshot: dict[str, object]) -> None:
        # Completion is checked only after the final gate has been passed.
        if self.status.completed or not self.gate_targets:
            return

        if self.status.current_gate_index < len(self.gate_targets):
            return

        goal_position = self.gate_targets[-1].center_world
        rov_position = np.array(snapshot["position"], dtype=float)
        if float(np.linalg.norm(rov_position - goal_position)) <= self.status.goal_radius_m:
            self.status.completed = True
            self.status.completion_time_s = float(snapshot["time_s"])

    def _update_energy(self, snapshot: dict[str, object]) -> None:
        # Integrate positive thruster power over time as a simple energy-consumption metric.
        time_s = float(snapshot["time_s"])
        if self._last_time_s is None:
            self._last_time_s = time_s
            return

        dt = max(time_s - self._last_time_s, 0.0)
        positive_power = max(float(snapshot["thruster_power_w"]), 0.0)
        positive_angular_power = max(float(snapshot["thruster_angular_power_w"]), 0.0)
        self.status.total_energy_consumed_j += (positive_power + positive_angular_power) * dt
        self._last_time_s = time_s

    def _register_impacts(self, contact_points: list[tuple], time_s: float) -> None:
        # Contact points come from Bullet; we keep only new contacts above a force threshold.
        current_ids: set[int] = set()
        for contact in contact_points:
            other_body_id = int(contact[2])
            normal_force = float(contact[9])
            if other_body_id < 0 or normal_force < 8.0:
                continue
            current_ids.add(other_body_id)
            if other_body_id in self._active_contact_ids:
                continue
            self.status.impact_events.append(
                {
                    "time_s": round(time_s, 5),
                    "other_body_id": other_body_id,
                    "normal_force_n": round(normal_force, 5),
                    "gate_index": self.status.current_gate_index,
                    "collision_type": "wall" if other_body_id in self.wall_ids else "obstacle",
                }
            )
        self._active_contact_ids = current_ids

    def write_logs(self) -> None:
        """Write the mission summary and impact-event log files."""
        self.summary_path.parent.mkdir(parents=True, exist_ok=True)
        self.impact_log_path.parent.mkdir(parents=True, exist_ok=True)

        with self.summary_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(
                [
                    "gates_passed",
                    "gates_total",
                    "mission_started_s",
                    "time_to_completion_s",
                    "completed",
                    "total_energy_consumed_j",
                    "total_points_captured",
                    "unique_seabed_cells",
                    "scanning_coverage_pct",
                    "mapping_efficiency_points_per_j",
                    "impact_event_count",
                    "wall_collision_count",
                ]
            )
            writer.writerow(
                [
                    len(self.status.gate_pass_times),
                    len(self.gate_targets),
                    "" if self.status.start_time_s is None else f"{self.status.start_time_s:.5f}",
                    ""
                    if self.status.completion_time_s is None
                    else f"{(self.status.completion_time_s - (self.status.start_time_s or 0.0)):.5f}",
                    str(self.status.completed),
                    f"{self.status.total_energy_consumed_j:.5f}",
                    self.total_points_captured,
                    len(self.unique_seabed_cells),
                    f"{self.scanning_coverage_pct():.5f}",
                    f"{self.mapping_efficiency():.8f}",
                    len(self.status.impact_events),
                    sum(1 for event in self.status.impact_events if event["collision_type"] == "wall"),
                ]
            )

        with self.impact_log_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "time_s",
                    "other_body_id",
                    "normal_force_n",
                    "gate_index",
                    "collision_type",
                ],
            )
            writer.writeheader()
            for event in self.status.impact_events:
                writer.writerow(event)


def automatic_obstacle_avoidance(
    snapshot: dict[str, object],
    scan_result: Optional[dict[str, object]],
    target_depth: float,
    gate_target: Optional[GateTarget],
    path_guidance: PathGuidance,
) -> dict[str, float]:
    """Simple layered autopilot.

    Priority order:
    1. Safety from active-sonar obstacles
    2. Pull back toward the path if crosstrack error grows
    3. Continue heading toward the next waypoint/gate
    """
    if not scan_result:
        return {
            "surge": 0.20,
            "sway": 0.0,
            "heave": 0.0,
            "roll": 0.0,
            "pitch": 0.0,
            "yaw": 0.0,
        }

    position = np.array(snapshot["position"], dtype=float)
    quaternion = np.array(snapshot["quaternion"], dtype=float)
    rotation_world_from_body = np.array(
        p.getMatrixFromQuaternion(quaternion), dtype=float
    ).reshape(3, 3)
    rotation_body_from_world = rotation_world_from_body.T
    local_velocity = np.array(snapshot["linear_velocity_local"], dtype=float)
    angular_velocity = np.array(snapshot["angular_velocity_local"], dtype=float)
    euler = np.array(snapshot["euler"], dtype=float)

    heading_vector_world = path_guidance.heading_vector_world
    heading_distance = float(np.linalg.norm(heading_vector_world))
    if heading_distance > 1e-6:
        heading_direction_local = rotation_body_from_world @ (heading_vector_world / heading_distance)
    else:
        heading_direction_local = np.array([1.0, 0.0, 0.0], dtype=float)
    crosstrack_vector_local = rotation_body_from_world @ path_guidance.crosstrack_vector_world

    # First compute the vertical tracking needed for the target depth/path.
    heave_error = target_depth - float(position[2])
    heave_tracking = float(np.clip(1.4 * heave_error - 0.45 * local_velocity[2], -0.45, 0.45))

    if gate_target is not None:
        # If a gate exists, center on it more aggressively than on the generic path alone.
        sway_centering = np.clip(0.95 * heading_direction_local[1] - 0.20 * local_velocity[1], -0.85, 0.85)
        heave_tracking += float(np.clip(0.70 * heading_direction_local[2], -0.25, 0.25))
    else:
        sway_centering = float(np.clip(0.80 * heading_direction_local[1], -0.60, 0.60))
        heave_tracking += float(np.clip(0.55 * heading_direction_local[2], -0.20, 0.20))

    sonar_max_range = float(scan_result["sonar_max_range"])
    detection_threshold = float(scan_result["detection_threshold"])
    forward_clearance = float(scan_result["forward_m"])
    backward_clearance = float(scan_result["backward_m"])
    left_clearance = float(scan_result["left_m"])
    right_clearance = float(scan_result["right_m"])
    up_clearance = float(scan_result["up_m"])
    down_clearance = float(scan_result["down_m"])
    forward_confidence = float(scan_result.get("forward_confidence", 1.0))

    def smooth_limit(value: float, scale: float, limit: float) -> float:
        # tanh gives a smooth saturation instead of abrupt clipping, reducing vibration.
        return float(limit * math.tanh(value / max(scale, 1e-6)))

    # Clearance differences tell the controller where open space exists around the vehicle.
    clearance_delta_lr = (left_clearance - right_clearance) / max(sonar_max_range, 1.0)
    clearance_delta_ud = (up_clearance - down_clearance) / max(sonar_max_range, 1.0)
    sway_repulsion = smooth_limit(1.2 * clearance_delta_lr, 0.45, 0.75)
    vertical_repulsion = smooth_limit(2.1 * clearance_delta_ud, 0.30, 0.50)
    surge_blockage = np.clip((1.7 - forward_clearance) / 1.7, -0.4, 1.0)
    aft_push = np.clip((1.2 - backward_clearance) / 1.2, 0.0, 0.4)
    confidence_weight = 0.55 + 0.45 * forward_confidence
    obstacle_ahead = forward_clearance < (1.15 * detection_threshold * confidence_weight)
    crosstrack_error = float(path_guidance.crosstrack_error_m)
    crosstrack_excess = max(crosstrack_error - 0.5, 0.0)
    path_pull_gain = np.clip(crosstrack_excess / 0.8, 0.0, 1.0)
    path_sway = np.clip(1.05 * crosstrack_vector_local[1] - 0.30 * local_velocity[1], -0.85, 0.85)
    path_heave = np.clip(0.95 * crosstrack_vector_local[2] - 0.28 * local_velocity[2], -0.70, 0.70)

    if obstacle_ahead:
        # If something is ahead, prioritize moving toward whichever vertical side is freer.
        vertical_bias = (down_clearance - up_clearance) / max(detection_threshold, 1e-6)
        vertical_repulsion = smooth_limit(3.2 * vertical_bias, 0.22, 0.75)
        sway_command = np.clip(1.22 * sway_repulsion + 0.45 * path_pull_gain * path_sway, -0.85, 0.85)
        heave = float(
            np.clip(
                1.60 * vertical_repulsion + 0.50 * path_pull_gain * path_heave - 0.20 * local_velocity[2],
                -0.80,
                0.80,
            )
        )
    else:
        sway_command = np.clip(
            0.45 * sway_centering + 0.80 * path_pull_gain * path_sway + 0.35 * sway_repulsion,
            -0.85,
            0.85,
        )
        heave = float(
            np.clip(
                0.55 * heave_tracking + 0.85 * path_pull_gain * path_heave + 0.30 * vertical_repulsion,
                -0.65,
                0.65,
            )
        )

    # Yaw turns the nose toward the path direction; pitch helps the ROV lean into vertical motion.
    yaw_command = float(
        np.clip(
            0.85 * math.atan2(heading_direction_local[1], max(heading_direction_local[0], 0.15))
            - 0.10 * angular_velocity[2],
            -0.40,
            0.40,
        )
    )

    alignment = float(np.clip(heading_direction_local[0], -1.0, 1.0))
    turn_penalty = np.clip(abs(heading_direction_local[1]) + 0.45 * abs(heading_direction_local[2]), 0.0, 1.0)
    obstacle_penalty = np.clip(
        (detection_threshold - forward_clearance) / max(detection_threshold, 1e-6), 0.0, 1.0
    )
    crosstrack_penalty = np.clip(crosstrack_error / 1.4, 0.0, 1.0)
    if forward_clearance < 0.9:
        # If something is very close, briefly reverse to create room.
        surge = -0.18
    elif forward_clearance < 1.4:
        # In a tight space, move forward slowly and carefully.
        surge = 0.08
    else:
        # In open water, surge depends on alignment, turn demand, obstacles, and path error.
        surge = 0.14 + 0.42 * max(alignment, 0.0)
        surge *= 1.0 - 0.55 * turn_penalty
        surge *= 1.0 - 0.65 * obstacle_penalty
        surge *= 1.0 - 0.30 * crosstrack_penalty
        surge += 0.05 * aft_push
        surge += 0.04 * np.clip(heading_distance / 3.0, 0.0, 1.0)
        surge = float(np.clip(surge - 0.16 * surge_blockage, 0.06, 0.60))

    if int(snapshot.get("sector_index", 1)) == 4 and surge > 0.0:
        # In the Abyssal zone we deliberately slow down to increase mapping density.
        surge *= 0.5

    pitch_command = float(
        np.clip(
            0.45 * heave + 0.30 * heading_direction_local[2] - 0.22 * euler[1] - 0.10 * angular_velocity[1],
            -0.35,
            0.35,
        )
    )

    return {
        "surge": surge,
        "sway": float(sway_command),
        "heave": heave,
        "roll": float(
            np.clip(
                -1.6 * euler[0] - 0.35 * angular_velocity[0],
                -0.35,
                0.35,
            )
        ),
        "pitch": pitch_command,
        "yaw": yaw_command,
    }


def scripted_input(sim_time: float) -> dict[str, float]:
    """Fallback motion used only before the first physics snapshot exists."""
    return {
        "surge": 0.32 * math.sin(0.55 * sim_time),
        "sway": 0.18 * math.sin(0.35 * sim_time + 0.4),
        "heave": 0.12 * math.sin(0.45 * sim_time + 1.1),
        "roll": 0.0,
        "pitch": 0.0,
        "yaw": 0.0,
    }


def build_status_text(
    mode: str,
    snapshot: dict[str, object],
    tuning: RuntimeTuning,
    params: ROVParameters,
    scan_result: Optional[dict[str, object]],
    gate_target: Optional[GateTarget],
    mission_status: MissionStatus,
    total_gates: int,
    path_guidance: PathGuidance,
) -> str:
    """Build the multiline overlay shown in the GUI."""
    position = snapshot["position"]
    linear_velocity = snapshot["linear_velocity_local"]
    forward_clearance = "n/a" if not scan_result else f"{scan_result['forward_m']:.2f} m"
    geometry = params.geometry
    gate_label = "none" if gate_target is None else (
        f"({gate_target.center_world[0]:+.2f}, {gate_target.center_world[1]:+.2f}, {gate_target.center_world[2]:+.2f})"
    )
    restoring = snapshot["restoring_moment_local"]
    forward_confidence = "n/a" if not scan_result else f"{scan_result.get('forward_confidence', 0.0):.2f}"
    completion_label = (
        "not-complete"
        if mission_status.completion_time_s is None
        else f"{mission_status.completion_time_s - (mission_status.start_time_s or 0.0):.2f}s"
    )
    return (
        f"Mode: {mode}   Forward: {forward_clearance}\n"
        f"Next Gate: {gate_label}   Sonar: {tuning.sonar_max_range:.1f} m / {tuning.num_lasers} rays @ {tuning.ping_frequency_hz:.1f} Hz\n"
        f"Progress: {mission_status.current_gate_index}/{total_gates}   Crosstrack: {path_guidance.crosstrack_error_m:.2f} m   Sector: {snapshot['sector_name']}\n"
        f"Cylinder r={geometry.radius:.2f} m  L={geometry.length:.2f} m  CoM z={geometry.com_offset_z:+.3f} m\n"
        f"Ixx={geometry.inertia_xx:.3f}  Iyy={geometry.inertia_yy:.3f}  Izz={geometry.inertia_zz:.3f}\n"
        f"pos=({position[0]:+.2f}, {position[1]:+.2f}, {position[2]:+.2f}) m  "
        f"u,v,w=({linear_velocity[0]:+.2f}, {linear_velocity[1]:+.2f}, {linear_velocity[2]:+.2f}) m/s\n"
        f"Forward Sonar Confidence: {forward_confidence}\n"
        f"Temp={snapshot['temperature_c']:.2f} C  Lux={snapshot['luminosity_lux']:.1f}  O2={snapshot['oxygen_pct']:.1f}%  pH={snapshot['ph']:.2f}\n"
        f"Tank Clearances F/B/L/R/U/D=({snapshot['clearance_front_m']:.2f}, {snapshot['clearance_back_m']:.2f}, "
        f"{snapshot['clearance_left_m']:.2f}, {snapshot['clearance_right_m']:.2f}, {snapshot['clearance_top_m']:.2f}, "
        f"{snapshot['clearance_bottom_m']:.2f}) m\n"
        f"mass={geometry.mass:.2f} kg  rho={tuning.rho:.1f}  Cd={tuning.drag_coefficient:.2f}  "
        f"B={tuning.buoyancy_multiplier:.2f}  CB-CG=+0.050 m  "
        f"M_rest=({restoring[0]:+.2f}, {restoring[1]:+.2f}, {restoring[2]:+.2f}) Nm\n"
        f"Impacts: {len(mission_status.impact_events)}  Energy Used: {mission_status.total_energy_consumed_j:.1f} J  "
        f"Completion: {completion_label}"
    )


def build_environment_text(snapshot: dict[str, object]) -> str:
    """Compact environmental overlay shown near the top of the GUI."""
    seabed_point = snapshot["seabed_point_world"]
    seabed_label = "none"
    if not np.isnan(seabed_point[0]):
        seabed_label = f"({seabed_point[0]:+.1f}, {seabed_point[1]:+.1f}, {seabed_point[2]:+.1f})"
    return (
        f"{snapshot['sector_name']}  Temp {snapshot['temperature_c']:.2f} C\n"
        f"Lux {snapshot['luminosity_lux']:.1f}  O2 {snapshot['oxygen_pct']:.1f}%  pH {snapshot['ph']:.2f}\n"
        f"Last Seabed Hit: {seabed_label}"
    )


def parse_args() -> argparse.Namespace:
    """Command-line options for GUI demos and headless validation runs."""
    parser = argparse.ArgumentParser(
        description="PyBullet cylindrical underwater ROV simulator for mathematical validation."
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run in DIRECT mode for smoke tests or CSV generation.",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=45.0,
        help="Maximum runtime in seconds for headless runs. GUI mode exits with Esc.",
    )
    parser.add_argument(
        "--csv-path",
        type=Path,
        default=Path("outputs") / "energy_log.csv",
        help="Path for the CSV energy log.",
    )
    parser.add_argument(
        "--plot-path",
        type=Path,
        default=Path("outputs") / "validation_dashboard.png",
        help="PNG output for the live validation dashboard.",
    )
    parser.add_argument(
        "--mission-summary-path",
        type=Path,
        default=Path("outputs") / "mission_summary.csv",
        help="CSV summary for mission completion metrics.",
    )
    parser.add_argument(
        "--seabed-map-path",
        type=Path,
        default=Path("outputs") / "seabed_map.csv",
        help="CSV log of seabed sonar mapping points.",
    )
    parser.add_argument(
        "--impact-log-path",
        type=Path,
        default=Path("outputs") / "impact_events.csv",
        help="CSV log of impact events against walls or obstacles.",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Skip dashboard export at shutdown.",
    )
    return parser.parse_args()


def main() -> None:
    """Set up the world, run the main loop, and save logs/plots on exit."""
    args = parse_args()
    connection_mode = p.DIRECT if args.headless else p.GUI
    if args.headless:
        client_id = p.connect(connection_mode)
    else:
        client_id = p.connect(connection_mode, options="--width=1540 --height=920")

    if not args.headless:
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 1, physicsClientId=client_id)
        p.resetDebugVisualizerCamera(
            cameraDistance=4.8,
            cameraYaw=32,
            cameraPitch=-22,
            cameraTargetPosition=[0.0, 0.0, -1.0],
            physicsClientId=client_id,
        )

    params = ROVParameters()
    # These defaults match the slider defaults so the GUI and physics engine start consistent.
    initial_tuning = RuntimeTuning(
        rho=params.hydro.rho,
        drag_coefficient=params.hydro.drag_coefficient,
        buoyancy_multiplier=params.hydro.buoyancy_multiplier,
        sonar_max_range=6.0,
        num_lasers=5,
        detection_threshold=2.0,
        ping_frequency_hz=10.0,
        sonar_noise_level_m=0.04,
        zone_transition_smoothing_m=2.0,
    )

    tank_bounds, wall_ids = world_setup(client_id)
    _, gate_targets = generate_underwater_course(client_id)
    if gate_targets:
        # Spawn slightly before the first gate, facing down the course.
        first_gate = gate_targets[0]
        params.start_position = first_gate.center_world + np.array([-0.9, 0.0, 0.0], dtype=float)
        params.start_rpy = np.array([0.0, 0.0, first_gate.yaw], dtype=float)
    body_id = create_rov(client_id, params, Path(__file__).with_name("assets") / "rov.urdf")
    controller = ROVController()
    engine = WaterPhysicsEngine(client_id, body_id, controller, params, tank_bounds, args.csv_path)
    active_sonar = ActiveSonar(client_id, body_id, engine.bullet_lock, seabed_z=tank_bounds.z_min)
    seabed_map_logger = SeabedMapLogger(client_id, engine.bullet_lock, args.seabed_map_path)
    mission_manager = MissionManager(
        gate_targets,
        args.mission_summary_path,
        args.impact_log_path,
        wall_ids,
        params.start_position,
        tank_bounds,
    )
    dashboard = SimulationDashboard(enabled=not args.headless)
    dashboard.start()

    parameter_panel = ParameterPanel(client_id, initial_tuning) if not args.headless else None
    overlay_id = -1
    environment_overlay_id = -1
    mode = AUTOMATIC_MODE if args.headless else MANUAL_MODE
    latest_scan: Optional[dict[str, object]] = None
    last_sonar_ping_time = -1.0
    target_depth = float(params.start_position[2] + params.geometry.com_offset_z)

    engine.update_runtime_tuning(initial_tuning)
    seabed_map_logger.start()
    engine.start()

    try:
        main_start = time.perf_counter()
        while True:
            if args.headless and time.perf_counter() - main_start >= args.duration:
                break
            elapsed = time.perf_counter() - main_start
            tuning = parameter_panel.read() if parameter_panel is not None else initial_tuning
            engine.update_runtime_tuning(tuning)

            events = p.getKeyboardEvents(physicsClientId=client_id) if not args.headless else {}
            if not args.headless and key_triggered(events, ESCAPE_KEY):
                break
            mode = handle_mode_switch(mode, events)
            snapshot = engine.get_state_snapshot()

            if snapshot:
                # Guidance is always based on the latest vehicle pose.
                gate_target = mission_manager.current_gate_target()
                path_guidance = mission_manager.current_path_guidance(
                    np.array(snapshot["position"], dtype=float)
                )
                engine.update_path_metrics(path_guidance.crosstrack_error_m)
                sonar_ping_interval = 1.0 / max(tuning.ping_frequency_hz, 1e-6)
                if elapsed - last_sonar_ping_time >= sonar_ping_interval:
                    # Lower ping rates deliberately make the vehicle "blind" for longer between updates.
                    latest_scan = active_sonar.scan(
                        snapshot,
                        draw_debug=not args.headless,
                        sonar_max_range=tuning.sonar_max_range,
                        num_lasers=tuning.num_lasers,
                        detection_threshold=tuning.detection_threshold,
                        sonar_noise_level_m=tuning.sonar_noise_level_m,
                    )
                    last_sonar_ping_time = elapsed

                if latest_scan is not None:
                    engine.update_sonar_metrics(
                        latest_scan.get("forward_confidence", 0.0),
                        latest_scan.get("mean_confidence", 0.0),
                    )
                    seabed_point_world = latest_scan.get("seabed_point_world")
                    if seabed_point_world is not None:
                        seabed_point_world = np.array(seabed_point_world, dtype=float)
                        seabed_environment = get_environmental_data(
                            seabed_point_world,
                            zone_transition_smoothing_m=tuning.zone_transition_smoothing_m,
                        )
                        engine.update_seabed_mapping(seabed_point_world, seabed_environment)
                        seabed_map_logger.record_hit(
                            float(snapshot["time_s"]),
                            seabed_point_world,
                            seabed_environment,
                            draw_debug=not args.headless,
                        )
                        mission_manager.register_seabed_hit(seabed_point_world)
                    else:
                        engine.update_seabed_mapping(None, None)

                if args.headless or mode == AUTOMATIC_MODE:
                    # Automatic mode uses the autopilot to generate 6-DOF commands.
                    command = automatic_obstacle_avoidance(
                        snapshot, latest_scan, target_depth, gate_target, path_guidance
                    )
                else:
                    # Manual mode maps the current keyboard state directly into body-axis commands.
                    command = manual_keyboard_command(events)

                engine.update_command_axes(command)
                with engine.bullet_lock:
                    contact_points = p.getContactPoints(bodyA=body_id, physicsClientId=client_id)
                mission_manager.update(snapshot, contact_points)
                dashboard.push_snapshot(snapshot)
                dashboard.maybe_refresh()

                if not args.headless:
                    # The text overlay acts like a compact live dashboard during demos.
                    caption = build_status_text(
                        mode,
                        snapshot,
                        tuning,
                        params,
                        latest_scan,
                        gate_target,
                        mission_manager.status,
                        len(gate_targets),
                        path_guidance,
                    )
                    with engine.bullet_lock:
                        overlay_id = p.addUserDebugText(
                            caption,
                            [0.55, -0.95, 0.35],
                            textColorRGB=[0.95, 0.95, 0.95],
                            textSize=1.05,
                            replaceItemUniqueId=overlay_id,
                            physicsClientId=client_id,
                        )
                        environment_overlay_id = p.addUserDebugText(
                            build_environment_text(snapshot),
                            [tank_bounds.x_min + 2.0, tank_bounds.y_min + 0.20, tank_bounds.z_max - 0.15],
                            textColorRGB=[0.92, 0.96, 0.98],
                            textSize=1.15,
                            replaceItemUniqueId=environment_overlay_id,
                            physicsClientId=client_id,
                        )
            else:
                # This startup fallback only runs before the physics thread publishes its first state.
                engine.update_command_axes(scripted_input(elapsed))

            time.sleep(1.0 / 60.0)
    except KeyboardInterrupt:
        pass
    finally:
        engine.stop()
        mission_manager.write_logs()
        if not args.no_plot:
            dashboard.save(args.plot_path)
        dashboard.close()
        p.disconnect(client_id)


if __name__ == "__main__":
    main()
