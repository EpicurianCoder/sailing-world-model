import math


class SailingPhysics:
    def __init__(self, drag_coeff=0.05, wind_hull_base=0.5, stress_threshold=100.0):
        # Set the parameters for the physcis engine
        self.C_drag = drag_coeff
        self.W_hull = wind_hull_base
        self.F_max = stress_threshold

    def Step(self, current_state, controller_target_actions, wind_dir, wind_strength):
        """ai_intent + physics -> new state"""
        # mechanical limits
        actual_boat_heading, actual_sail_angle, actual_sail_size = self._process_actions(
            current_state,
            controller_target_actions
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

    def _process_actions(self, current_state, controller_target_actions):
        MAX_TURN_RATE = 10.0
        MAX_SAIL_TURN_RATE = 10.0
        MAX_SAIL_SIZE_RATE = 0.2

        SAIL_ANGLE_LIMIT = 90.0
        MIN_SAIL_SIZE = 3.0
        MAX_SAIL_SIZE = 6.0

        # Continuous Steering: Action [-1.0, 1.0] directly scales the max turn rate
        delta_boat = controller_target_actions['steer_axis'] * MAX_TURN_RATE
        new_boat_angle = (current_state['boat_angle'] + delta_boat) % 360.0

        # Continuous Sail Angle
        delta_sail = controller_target_actions['sail_trim_axis'] * MAX_SAIL_TURN_RATE
        new_sail_angle = max(-SAIL_ANGLE_LIMIT, min(current_state['sail_angle'] + delta_sail, SAIL_ANGLE_LIMIT))

        # Sail Size - Absolute
        diff_size = controller_target_actions['sail_size'] - current_state['sail_size']
        delta_size = max(-MAX_SAIL_SIZE_RATE, min(diff_size, MAX_SAIL_SIZE_RATE))
        new_sail_size = max(MIN_SAIL_SIZE, min(current_state['sail_size'] + delta_size, MAX_SAIL_SIZE))

        return new_boat_angle, new_sail_angle, new_sail_size    

    def _process_actions_old(self, current_state, controller_target_actions):
        # Maximum increments allowed
        MAX_TURN_RATE = 10.0
        MAX_SAIL_TURN_RATE = 10.0
        MAX_SAIL_SIZE_RATE = 0.2

        # mechanical boundaries
        SAIL_ANGLE_LIMIT = 90.0
        MIN_SAIL_SIZE = 3.0
        MAX_SAIL_SIZE = 6.0

        # Update Boat Angle
        diff_boat = (controller_target_actions['boat_angle'] - current_state['boat_angle'] + 180.0) % 360.0 - 180.0
        delta_boat = max(-MAX_TURN_RATE, min(diff_boat, MAX_TURN_RATE))
        new_boat_angle = (current_state['boat_angle'] + delta_boat) % 360.0

        # Update Sail Angle
        diff_sail = controller_target_actions['sail_angle'] - current_state['sail_angle']
        delta_sail = max(-MAX_SAIL_TURN_RATE, min(diff_sail, MAX_SAIL_TURN_RATE))
        new_sail_angle = max(-SAIL_ANGLE_LIMIT, min(current_state['sail_angle'] + delta_sail, SAIL_ANGLE_LIMIT))

        # Update Sail Size
        diff_size = controller_target_actions['sail_size'] - current_state['sail_size']
        delta_size = max(-MAX_SAIL_SIZE_RATE, min(diff_size, MAX_SAIL_SIZE_RATE))
        new_sail_size = max(MIN_SAIL_SIZE, min(current_state['sail_size'] + delta_size, MAX_SAIL_SIZE))

        return new_boat_angle, new_sail_angle, new_sail_size

    def _calculate_physics(self, boat_heading, sail_angle, sail_size, wind_dir, wind_strength, current_velocity):

        # Relative Wind (-180 to 180)
        # Positive = Wind from Starboard (Right), Negative = Wind from Port (Left)
        theta_rel_signed = (wind_dir - boat_heading + 180) % 360 - 180
        theta_rel_abs = abs(theta_rel_signed)

        # Optimal Sail Angle
        # The sail must be pushed to the opposite side of the incoming wind!
        theta_opt = -(theta_rel_signed / 2.0)
        
        # Detla to optimal sail angle
        delta_sail = abs(sail_angle - theta_opt)

        # Forces
        if theta_rel_abs < 45.0:
            # No-Go Zone -> CAPSIZE!!!
            cos_rel = math.cos(math.radians(theta_rel_abs))
            f_drive = -wind_strength * (self.W_hull + sail_size) * cos_rel
        else:
            # Forward drive
            # When (delta_sail > 90) the sail is on the WRONG side
            # Forward drive becomes 0
            trim_efficiency = max(0.0, 1.0 - (delta_sail / 45.0))
            f_drive = wind_strength * sail_size * trim_efficiency

        # Apply Drag
        direction_sign = 1 if current_velocity >= 0 else -1

        # BIG drag multiplier for moving backwards
        drag_multiplier = 1.0 if current_velocity >= 0 else 20.0
        f_drag = (self.C_drag * drag_multiplier) * (current_velocity ** 2) * direction_sign

        # Mass variable to influence acceleration
        mass = 5.0
        net_force = f_drive - f_drag

        # Velocity (F = MA)
        new_vel = current_velocity + (net_force / mass)

        # ABSOLUTE SPEED LIMITS
        MAX_FORWARD_SPEED = 3.0
        MAX_REVERSE_SPEED = -1.0
        new_vel = max(MAX_REVERSE_SPEED, min(new_vel, MAX_FORWARD_SPEED))

        # Failure States
        # SINE is used here:
        #   Perfect trim (0 deg) = 0 stress.
        #   Backed sail (90 deg) = Max stress.
        sin_delta = math.sin(math.radians(delta_sail))

        # Wind from wrong side and stress goes up
        f_rig = wind_strength * sail_size * abs(sin_delta)

        stress_multiplier = (1.0 + math.cos(math.radians(theta_rel_abs))) / 2.0
        f_stress = f_rig * stress_multiplier

        # Capsize if stress exceeds max force
        if f_stress > self.F_max:
            is_capsized = True
        else:
            is_capsized = False

        return new_vel, is_capsized

    def _update_position(self, current_x, current_y, boat_heading, velocity):
        # Convert heading to radians for math functions
        heading_rad = math.radians(boat_heading)

        dt = 0.25
        scaled_velocity = velocity * dt

        # Calculate coordinate deltas
        # Sine for X (East/West), Cosine for Y (North/South)
        delta_x = scaled_velocity * math.sin(heading_rad)
        delta_y = scaled_velocity * math.cos(heading_rad)

        # Update coordinates
        new_x = current_x + delta_x
        new_y = current_y + delta_y

        return new_x, new_y

    def check_terminal_states(self, new_x, new_y, is_capsized):
        if is_capsized:
            return True, "capsized"

        if new_y >= 256.0:
            return True, "success"

        if new_x < 0.0 or new_x > 256.0 or new_y < 0.0:
            return True, "out_of_bounds"

        # The simulation continues!!!
        return False, "active"
