import numpy as np
from scipy.ndimage import distance_transform_edt
import math
import config


def dist_to_land(land_channel, boat_channel):
    land_arr = np.asarray(land_channel)
    boat_arr = np.asarray(boat_channel)

    if not np.any(land_arr == 1) or not np.any(boat_arr == 1):
        return float('inf')

    dist_map = distance_transform_edt(land_arr == 0)
    min_distance = float(np.min(dist_map[boat_arr == 1]))

    return min_distance


def los_finish_line_bearing(boat_xy, boat_bearing, land_channel, map_size=config.MAP_SIZE):
    land_arr = np.asarray(land_channel)
    boat_x, boat_y = boat_xy
    bearing_rad = math.radians(boat_bearing)

    # direction for the ray
    dir_x = math.sin(bearing_rad)
    dir_y = math.cos(bearing_rad)

    if dir_y <= 0:
        return False

    # Step forward pixel-by-pixel along the ray
    for step in range(1, int(map_size * 2)):
        curr_x = boat_x + dir_x * step
        curr_y = boat_y + dir_y * step

        if curr_y >= map_size:
            return True

        if curr_x < 0 or curr_x >= map_size:
            return False

        grid_x = int(curr_x)
        grid_y = int(map_size - curr_y)

        grid_x = max(0, min(map_size - 1, grid_x))
        grid_y = max(0, min(map_size - 1, grid_y))

        # Did we hit an island?
        if land_arr[grid_y, grid_x] == 1:
            return False

    return False


def line_of_sight_dist(boat_xy, boat_bearing, land_channel, max_distance=256.0, map_size=config.MAP_SIZE):
    land_arr = np.asarray(land_channel)
    bearing_rad = math.radians(boat_bearing)
    boat_x, boat_y = boat_xy

    # direction for the ray
    dir_x = math.sin(bearing_rad)
    dir_y = math.cos(bearing_rad)

    # Step along the ray
    for step in range(1, int(max_distance) + 1):
        curr_x = boat_x + dir_x * step
        curr_y = boat_y + dir_y * step

        if curr_x < 0 or curr_x >= map_size or curr_y < 0 or curr_y >= map_size:
            return float(step)

        grid_x = int(curr_x)
        grid_y = int(map_size - curr_y)
        grid_x = max(0, min(map_size - 1, grid_x))
        grid_y = max(0, min(map_size - 1, grid_y))

        if land_arr[grid_y, grid_x] == 1:
            return float(step)

    return max_distance


def boat_abs_xy(boat_xy, map_size=config.MAP_SIZE, normalize=True):
    boat_x, boat_y = boat_xy
    if normalize:
        return (boat_x / map_size, boat_y / map_size)
    return (float(boat_x), float(boat_y))


def sector_land_density(land_channel, grid_size=3):
    land_arr = np.asarray(land_channel)
    height, width = land_arr.shape

    row_step = height // grid_size
    col_step = width // grid_size

    sector_densities = []

    for r in range(grid_size):
        for c in range(grid_size):
            r_start, r_end = r * row_step, (r + 1) * row_step
            c_start, c_end = c * col_step, (c + 1) * col_step

            sector = land_arr[r_start:r_end, c_start:c_end]
            density = float(np.mean(sector == 1))
            sector_densities.append(density)

    return sector_densities


def finish_delta(current_xy, start_xy=(128.0, 22.5), finish_xy=(128.0, 256.0), map_size=config.MAP_SIZE, normalize=True):
    curr_x, curr_y = current_xy
    init_x, init_y = start_xy
    fin_x, fin_y = finish_xy

    # Initial distance to finish
    initial_dist = math.sqrt((fin_x - init_x) ** 2 + (fin_y - init_y) ** 2)
    # Current distance to finish
    current_dist = math.sqrt((fin_x - curr_x) ** 2 + (fin_y - curr_y) ** 2)

    delta = initial_dist - current_dist

    if normalize:
        return delta / map_size  # Normalized relative to map size [e.g., -1.0 to 1.0]

    return delta
