import math


class SailingPhysics:
    """
    This happens at the VERY beginning of each loop, before the grid
    and visuals are rendered, and before the current_step is generated.
    It works on information provided from the previous step.

    drag_coeff: The amount of drag the boat experiences
    wind_hull_base: The amount the wind blows the physical boat vs the sail
    stress_threshold: The Force at which a catastrophic event will occur
    """
    def __init__(self, drag_coeff=0.05, wind_hull_base=0.5, stress_threshold=100.0):
        # Set the parameters for the physcis engine
        self.C_drag = drag_coeff
        self.W_hull = wind_hull_base
        self.F_max = stress_threshold

    def Step(self, current_state, ai_targets, wind_dir, wind_strength):
        """ai_intent + physics -> new state"""
        # mechanical limits
        actual_boat_heading, actual_sail_angle, actual_sail_size = self._process_actions(
            current_state,
            ai_targets
        )

        # Environmental Physics
        new_velocity, is_capsized = self._calculate_physics(
            actual_boat_heading,
            actual_sail_angle,
            actual_sail_size,
            wind_dir,
            wind_strength,
            current_state['velocity']
        )

        # Spatial Movement
        new_x, new_y = self._update_position(
            current_state['x'],
            current_state['y'],
            actual_boat_heading,
            new_velocity
        )

        # Package the new_state object
        new_state = {
            'x': new_x,
            'y': new_y,
            'boat_angle': actual_boat_heading,
            'sail_angle': actual_sail_angle,
            'sail_size': actual_sail_size,
            'velocity': new_velocity
        }

        return new_state, is_capsized

    def _process_actions(self, current_state, ai_targets):
        # Maximum increments (rates of change per frame)
        MAX_TURN_RATE = 10.0
        MAX_SAIL_TURN_RATE = 10.0
        MAX_SAIL_SIZE_RATE = 0.2

        # Absolute mechanical boundaries
        SAIL_ANGLE_LIMIT = 90.0
        MIN_SAIL_SIZE = 3.0
        MAX_SAIL_SIZE = 6.0

        # Update Boat Angle
        diff_boat = (ai_targets['boat_angle'] - current_state['boat_angle'] + 180.0) % 360.0 - 180.0
        delta_boat = max(-MAX_TURN_RATE, min(diff_boat, MAX_TURN_RATE))
        new_boat_angle = (current_state['boat_angle'] + delta_boat) % 360.0

        # Update Sail Angle (This uses a Linear path with boundaries)
        diff_sail = ai_targets['sail_angle'] - current_state['sail_angle']
        delta_sail = max(-MAX_SAIL_TURN_RATE, min(diff_sail, MAX_SAIL_TURN_RATE))
        new_sail_angle = max(-SAIL_ANGLE_LIMIT, min(current_state['sail_angle'] + delta_sail, SAIL_ANGLE_LIMIT))

        # Update Sail Size (Linear path with boundaries)
        diff_size = ai_targets['sail_size'] - current_state['sail_size']
        delta_size = max(-MAX_SAIL_SIZE_RATE, min(diff_size, MAX_SAIL_SIZE_RATE))
        new_sail_size = max(MIN_SAIL_SIZE, min(current_state['sail_size'] + delta_size, MAX_SAIL_SIZE))

        return new_boat_angle, new_sail_angle, new_sail_size

    def _calculate_physics(self, boat_heading, sail_angle, sail_size, wind_dir, wind_strength, current_velocity):
        
        # Relative wind angle (0 to 180 degrees)
        theta_rel = abs((wind_dir - boat_heading + 180) % 360 - 180)

        # Optimal sail angle and delta
        theta_opt = theta_rel / 2.0
        delta_sail = abs(sail_angle - theta_opt)

        # Forces and Movement
        if theta_rel < 45.0:
            # No-Go Zone: Blown backwards
            cos_rel = math.cos(math.radians(theta_rel))
            f_drive = -wind_strength * (self.W_hull + sail_size) * cos_rel
        else:
            # Sailing: Forward drive
            trim_efficiency = max(0.0, 1.0 - (delta_sail / 45.0))
            f_drive = wind_strength * sail_size * trim_efficiency

        # Apply Drag
        direction_sign = 1 if current_velocity >= 0 else -1
        f_drag = self.C_drag * (current_velocity ** 2) * direction_sign

        # Velocity Update
        new_vel = current_velocity + f_drive - f_drag

        # Failure States (Capsizing/Stress)
        cos_delta = math.cos(math.radians(delta_sail))
        f_rig = wind_strength * sail_size * cos_delta

        stress_multiplier = (1.0 + math.cos(math.radians(theta_rel))) / 2.0
        f_stress = f_rig * stress_multiplier

        is_capsized = f_stress > self.F_max

        return new_vel, is_capsized

    def _update_position(self, current_x, current_y, boat_heading, velocity):
        # Convert heading to radians for math functions
        heading_rad = math.radians(boat_heading)

        # Calculate coordinate deltas
        # Sine for X (East/West), Cosine for Y (North/South)
        delta_x = velocity * math.sin(heading_rad)
        delta_y = velocity * math.cos(heading_rad)

        # Update coordinates
        new_x = current_x + delta_x
        new_y = current_y + delta_y

        return new_x, new_y

    def check_terminal_states(self, new_x, new_y, is_capsized):
        """
        Evaluates the boat's position and physical status to determine
        if the simulation round should end, and for what reason.
        Returns: (is_terminal_boolean, status_string)
        """
        if is_capsized:
            return True, "capsized"

        # Check Success
        if new_y >= 256.0:
            # Crossed Northern edge (Successfully)
            if 64.0 <= new_x <= 192.0:
                return True, "success"
            else:
                # Crossed Northern edge (but missed the target zone)
                return True, "out_of_bounds"

        # Check Out of Bounds (East, West, or South)
        if new_x < 0.0 or new_x > 256.0 or new_y < 0.0:
            return True, "out_of_bounds"

        # If none of the above, the simulation continues
        return False, "active"
