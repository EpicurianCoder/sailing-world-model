from dataclasses import dataclass, asdict
import numpy as np
import math


@dataclass
class EnvironmentState:
    # Physcis
    boat_x: float
    boat_y: float
    boat_angle: float
    boat_speed: float
    sail_angle: float
    sail_size: float

    # Env
    wind_angle: float
    wind_strength: float
    theta_rel: float
    theta_opt: float

    # ground truths
    dist_to_land: float
    line_of_sight: float
    sector_densities: list

    def to_dict(self) -> dict:
        d = asdict(self)
        d['x'] = self.boat_x
        d['y'] = self.boat_y
        d['velocity'] = self.boat_speed
        return d

    def to_telemetry_vector(self) -> np.ndarray:
        # Normalize non-circular values
        norm_speed = (self.boat_speed - 1.0) / 2.0
        norm_sail_size = (self.sail_size - 4.5) / 1.5
        norm_dist_land = min(self.dist_to_land, 256.0) / 256.0
        norm_los = min(self.line_of_sight, 256.0) / 256.0

        # Convert circular angles to Radians
        boat_rad = math.radians(self.boat_angle)
        wind_rad = math.radians(self.wind_angle)
        rel_rad = math.radians(self.theta_rel)

        # Build continuous telemetry list
        telemetry = [
            norm_speed,
            math.sin(boat_rad),
            math.cos(boat_rad),
            self.sail_angle / 90.0,
            norm_sail_size,
            math.sin(wind_rad),
            math.cos(wind_rad),
            self.wind_strength / 20.0,
            math.sin(rel_rad),
            math.cos(rel_rad),
            self.theta_opt / 90.0,
            norm_dist_land,
            norm_los
        ]

        telemetry.extend(self.sector_densities)

        return np.array(telemetry, dtype=np.float32)

    def to_telemetry_vector_old(self) -> np.ndarray:
        norm_speed = (self.boat_speed - 1.0) / 2.0
        norm_sail_size = (self.sail_size - 4.5) / 1.5
        norm_dist_land = min(self.dist_to_land, 256.0) / 256.0
        norm_los = min(self.line_of_sight, 256.0) / 256.0

        telemetry = [
            norm_speed,
            self.boat_angle / 360.0,
            self.sail_angle / 90.0,
            norm_sail_size,
            self.wind_angle / 360.0,
            self.wind_strength / 20.0,
            self.theta_rel / 180.0,
            self.theta_opt / 90.0,
            norm_dist_land,
            norm_los
        ]

        telemetry.extend(self.sector_densities)

        return np.array(telemetry, dtype=np.float32)


@dataclass
class StepTransition:
    # Observations
    obs_current_grid: np.ndarray
    obs_current_state: EnvironmentState

    action: np.ndarray
    reward: float
    done: bool
    status: str
    obs_next_grid: np.ndarray
    obs_next_state: EnvironmentState
