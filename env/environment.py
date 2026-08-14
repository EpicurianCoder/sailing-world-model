import random
import numpy as np
# import math
from structures import StepTransition, EnvironmentState
from telemetry import dist_to_land, sector_land_density, los_finish_line_bearing
from feature_gen import FeatureGenerator


class SailingEnvironment:
    def __init__(self, physics_engine):
        self.physics = physics_engine
        self.land_grid, self.num_islands, self.coverage = FeatureGenerator.gen_land_feature()
        self.current_state = None

    def boat_start(self):
        # currently favourable for early training
        new_wind_angle = random.uniform(90.0, 270.0)
        wind_strength = random.uniform(5.0, 20.0)

        # Random Heading (Somewhat North though)
        new_boat_angle = random.uniform(-90.0, 90.0) % 360.0

        # Random Valid Sail Angle (withing constraints)
        new_sail_angle = random.uniform(-90.0, 90.0)

        # Telemetry Calc
        final_twa = (new_wind_angle - new_boat_angle + 180.0) % 360.0 - 180.0
        theta_opt = -(final_twa / 2.0)

        state = {
            'boat_x': random.randint(120, 136),
            'boat_y': random.randint(15, 30),
            'boat_angle': new_boat_angle,
            'boat_speed': random.uniform(1.0, 5.0),
            'sail_angle': new_sail_angle,
            'sail_size': random.uniform(3.0, 6.0),
            'wind_angle': new_wind_angle,
            'wind_strength': wind_strength,
            'theta_rel': final_twa,
            'theta_opt': theta_opt
        }

        return state

    # Initializes the environment state
    def reset(self):
        # gen starting physics and telemetry
        start_dict = self.boat_start()

        temp_boat_grid = FeatureGenerator.get_boat_feature_layer(
            start_dict['boat_x'], start_dict['boat_y'], start_dict['boat_angle']
        )

        # Calculate additional metrics
        boat_xy = (start_dict['boat_x'], start_dict['boat_y'])
        d_land = dist_to_land(self.land_grid, temp_boat_grid)
        los = los_finish_line_bearing(boat_xy, start_dict['boat_angle'], self.land_grid)
        sectors = sector_land_density(self.land_grid, grid_size=3)

        # initial EnvironmentState
        self.current_state = EnvironmentState(
            boat_x=start_dict['boat_x'],
            boat_y=start_dict['boat_y'],
            boat_angle=start_dict['boat_angle'],
            boat_speed=start_dict['boat_speed'],
            sail_angle=start_dict['sail_angle'],
            sail_size=start_dict['sail_size'],
            wind_angle=start_dict['wind_angle'],
            wind_strength=start_dict['wind_strength'],
            theta_rel=start_dict['theta_rel'],
            theta_opt=start_dict['theta_opt'],
            dist_to_land=d_land,
            line_of_sight=los,
            sector_densities=sectors
        )

        return self._build_3_channels(self.current_state), self.current_state

    def _calculate_reward(self, status, current_state, next_state, step_count) -> float:
        prev_grad = current_state.boat_y / 256.0
        curr_grad = next_state.boat_y / 256.0
        prog_reward = curr_grad - prev_grad

        if prog_reward > 0:
            prog_reward = 0.004
        elif prog_reward < 0:
            prog_reward = - 0.004

        # optimal wind capture reward
        delta_sail = abs(next_state.sail_angle - next_state.theta_opt)
        trim_efficiency = max(0.0, 1.0 - (delta_sail / 45.0))
        wind_reward = 0.001 * trim_efficiency

        # Line of Sight reward
        boat_xy = (next_state.boat_x, next_state.boat_y)
        boat_bearing = next_state.boat_angle
        has_los = los_finish_line_bearing(boat_xy, boat_bearing, self.land_grid)
        los_reward = 0.002 if has_los else 0.0

        # Proximity rewards / penalites
        prox_penalty = 0.0
        if next_state.dist_to_land <= 10.5:
            prox_penalty = -0.002
        elif next_state.dist_to_land <= 15.5:
            prox_penalty = -0.001

        # stress Penalty (I need to add this)
        stress_penalty = 0.0

        step_reward = prog_reward + wind_reward + los_reward + prox_penalty + stress_penalty

        # terminal rewards
        if status != "active":
            terminal_reward = self._calculate_terminal_reward(status, curr_grad, step_count)
            return terminal_reward

        return step_reward

    def _calculate_terminal_reward(self, status, final_gradient, step_count) -> float:
        if status == "success":
            speed_bonus = 0.05 * max(0.0, 1.0 - (step_count / 300.0))
            return 0.95 + speed_bonus

        elif status in ["crashed", "capsized", "out_of_bounds"]:
            return -1.0

        # If the run times out safely
        return self.scale_to_custom_range(final_gradient, 0.0, 1.0, 0.0, 0.04)

    # I got tired of writing weird scaling wrappers
    def scale_to_custom_range(self, value, orig_min, orig_max, target_min=0.0, target_max=1.0):
        normalized = (value - orig_min) / (orig_max - orig_min)
        scaled_value = target_min + (normalized * (target_max - target_min))
        return max(target_min, min(scaled_value, target_max))

    def _build_3_channels(self, state):
        boat_grid = FeatureGenerator.get_boat_feature_layer(
            state.boat_x,
            state.boat_y,
            state.boat_angle
        )

        land_np = np.asarray(self.land_grid, dtype=np.float32)
        boat_np = np.asarray(boat_grid, dtype=np.float32)

        height, width = land_np.shape
        gradient_1d = np.linspace(1.0, 0.0, height, dtype=np.float32) ** 2.0
        base_gradient = np.tile(gradient_1d[:, np.newaxis], (1, width))
        gradient_np = base_gradient * (1.0 - land_np)

        # Channel 0: Land
        # Channel 1: Boat
        # Channel 2: Gradient
        return np.stack([land_np, boat_np, gradient_np], axis=0)

    def step(self, controller_action_targets: dict, step_count: int = 0) -> StepTransition:
        current_state = self.current_state
        current_grid = self._build_3_channels(current_state)

        # Run Physics Engine
        next_physics_dict, is_capsized = self.physics.Step(
            current_state.to_dict(),
            controller_action_targets,
            current_state.wind_angle,
            current_state.wind_strength
        )

        # Determine Terminal States & Rewards
        is_terminal, status = self.physics.check_terminal_states(
            next_physics_dict['x'],
            next_physics_dict['y'],
            is_capsized
        )

        boat_xy = (next_physics_dict['x'], next_physics_dict['y'])

        # For extra distance calculations
        temp_boat_grid = FeatureGenerator.get_boat_feature_layer(
            next_physics_dict['x'],
            next_physics_dict['y'],
            next_physics_dict['boat_angle']
        )
        d_land = dist_to_land(self.land_grid, temp_boat_grid)

        # For crashing into land
        if not is_terminal and d_land == 0.0:
            is_terminal = True
            status = "crashed"

        los = los_finish_line_bearing(boat_xy, next_physics_dict['boat_angle'], self.land_grid)
        sectors = sector_land_density(self.land_grid, grid_size=3)

        next_state = EnvironmentState(
            boat_x=next_physics_dict['x'],
            boat_y=next_physics_dict['y'],
            boat_angle=next_physics_dict['boat_angle'],
            boat_speed=next_physics_dict['velocity'],
            sail_angle=next_physics_dict['sail_angle'],
            sail_size=next_physics_dict['sail_size'],
            wind_angle=current_state.wind_angle,
            wind_strength=current_state.wind_strength,
            theta_rel=(current_state.wind_angle - next_physics_dict['boat_angle'] + 180) % 360 - 180,
            theta_opt=-(((current_state.wind_angle - next_physics_dict['boat_angle'] + 180) % 360 - 180) / 2.0),
            dist_to_land=d_land,
            line_of_sight=los,
            sector_densities=sectors
        )
        reward = self._calculate_reward(status, current_state, next_state, step_count)

        next_grid = self._build_3_channels(next_state)

        transition = StepTransition(
            obs_current_grid=current_grid,
            obs_current_state=current_state,
            action=controller_action_targets,
            reward=reward,
            done=is_terminal,
            status=status,
            obs_next_grid=next_grid,
            obs_next_state=next_state
        )

        self.current_state = next_state

        return transition
