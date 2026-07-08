import unittest
from sailing_physics import SailingPhysics


class TestSailingPhysics(unittest.TestCase):

    def setUp(self):
        """Runs before every single test to give us a fresh physics engine."""
        # Using default values: drag=0.05, hull_wind=0.5, stress_max=100.0
        self.physics = SailingPhysics()

    def test_process_actions_clamping(self):
        """Tests if the mechanical turn rates and size limits are properly enforced."""
        current_state = {'boat_angle': 0.0, 'sail_angle': 0.0, 'sail_size': 4.0}

        # AI wants to turn 90 degrees instantly, rotate sail 90 degrees, and max out sail
        ai_targets = {'boat_angle': 90.0, 'sail_angle': 90.0, 'sail_size': 10.0}

        new_boat, new_sail, new_size = self.physics._process_actions(current_state, ai_targets)

        # Max turn rates are 10.0 per frame, sail size rate is 0.2
        self.assertEqual(new_boat, 10.0, "Boat turn rate should be clamped to 10 degrees")
        self.assertEqual(new_sail, 10.0, "Sail turn rate should be clamped to 10 degrees")
        self.assertAlmostEqual(new_size, 4.2, places=2, msg="Sail size increase should be clamped to 0.2")

    def test_calculate_physics_in_irons(self):
        """Tests if pointing dead into the wind blows the boat backwards."""
        # Wind from North (0), Boat pointing North (0) -> Relative angle is 0 (In Irons)
        new_vel, is_capsized = self.physics._calculate_physics(
            boat_heading=0.0, sail_angle=0.0, sail_size=3.0, 
            wind_dir=0.0, wind_strength=5.0, current_velocity=0.0
        )

        self.assertLess(new_vel, 0.0, "Boat should be pushed backwards (negative velocity)")
        self.assertFalse(is_capsized, "Boat should not capsize from head-on wind with small sail")

    def test_calculate_physics_sailing(self):
        """Tests if optimal sail trim generates forward momentum."""
        # Wind from East (90), Boat pointing North (0) -> Relative angle is 90
        # Optimal sail angle is 45. Let's set sail to exactly 45.
        new_vel, is_capsized = self.physics._calculate_physics(
            boat_heading=0.0, sail_angle=45.0, sail_size=5.0, 
            wind_dir=90.0, wind_strength=10.0, current_velocity=0.0
        )

        self.assertGreater(new_vel, 0.0, "Boat should gain forward positive velocity")

    def test_calculate_physics_capsizing(self):
        """Tests if severe directional stress triggers a capsize."""
        # Wind from East (90), Boat pointing North (0) -> Beam reach (side-on danger)
        # Gale force winds (50.0), fully extended sail (6.0), sail pulled tight (0.0) catching full wind
        new_vel, is_capsized = self.physics._calculate_physics(
            boat_heading=0.0, sail_angle=0.0, sail_size=6.0, 
            wind_dir=90.0, wind_strength=50.0, current_velocity=5.0
        )

        self.assertTrue(is_capsized, "Boat should capsize under extreme side-on wind stress")

    def test_update_position_trigonometry(self):
        """Tests if the heading correctly translates to X and Y grid movement."""
        # Test moving North (Heading 0)
        new_x, new_y = self.physics._update_position(128.0, 10.0, boat_heading=0.0, velocity=5.0)
        self.assertAlmostEqual(new_x, 128.0, places=2, msg="Moving North should not change X")
        self.assertAlmostEqual(new_y, 15.0, places=2, msg="Moving North should increase Y by velocity")

        # Test moving East (Heading 90)
        new_x, new_y = self.physics._update_position(128.0, 10.0, boat_heading=90.0, velocity=5.0)
        self.assertAlmostEqual(new_x, 133.0, places=2, msg="Moving East should increase X by velocity")
        self.assertAlmostEqual(new_y, 10.0, places=2, msg="Moving East should not change Y")

    def test_terminal_states(self):
        """Tests win conditions, out of bounds, and active states."""
        # 1. Success (Crossed Y=256 within 64-192 X bounds)
        is_term, status = self.physics.check_terminal_states(new_x=128.0, new_y=256.5, is_capsized=False)
        self.assertTrue(is_term)
        self.assertEqual(status, "success")

        # 2. Out of Bounds (Missed target zone)
        is_term, status = self.physics.check_terminal_states(new_x=50.0, new_y=257.0, is_capsized=False)
        self.assertTrue(is_term)
        self.assertEqual(status, "out_of_bounds")

        # 3. Out of Bounds (Off the sides)
        is_term, status = self.physics.check_terminal_states(new_x=-5.0, new_y=100.0, is_capsized=False)
        self.assertTrue(is_term)
        self.assertEqual(status, "out_of_bounds")

        # 4. Active (In the middle of the ocean)
        is_term, status = self.physics.check_terminal_states(new_x=128.0, new_y=128.0, is_capsized=False)
        self.assertFalse(is_term)
        self.assertEqual(status, "active")

if __name__ == '__main__':
    unittest.main()