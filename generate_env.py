import math
import random
from PIL import Image, ImageDraw
from noise import pnoise2

WATER_COLOR = (173, 216, 230)
LAND_COLOR = (210, 180, 140)
BOAT_COLOR = (255, 255, 255)
SAIL_COLOR = (255, 0, 255)
WIND_COLOR = (255, 0, 0)
SPEED_VEC_COLOR = (0, 0, 255)     # Blue
WIND_VEC_COLOR = (0, 255, 0)      # Green
OPT_SAIL_COLOR = (255, 255, 0)    # Yellow
RESULTANT_COLOR = (255, 165, 0)   # Orange
FWD_FORCE_COLOR = (0, 255, 255)   # Cyan


def draw_boat(draw, x, y, boat_angle, sail_angle, sail_size, map_size=256):
    screen_y = map_size - y
    boat_rad = math.radians(boat_angle)
    sail_rad = math.radians(boat_angle + sail_angle + 180)

    # Hull
    front_x = x + 3 * math.sin(boat_rad)
    front_y = screen_y - 3 * math.cos(boat_rad)
    back_x = x - 3 * math.sin(boat_rad)
    back_y = screen_y + 3 * math.cos(boat_rad)
    draw.line([(back_x, back_y), (front_x, front_y)], fill=BOAT_COLOR, width=1)

    # Sail
    pivot_x = x + 2 * math.sin(boat_rad)
    pivot_y = screen_y - 2 * math.cos(boat_rad)
    sail_end_x = pivot_x + sail_size * math.sin(sail_rad)
    sail_end_y = pivot_y - sail_size * math.cos(sail_rad)
    draw.line([(pivot_x, pivot_y), (sail_end_x, sail_end_y)], fill=SAIL_COLOR, width=1)


def draw_wind_arrow(draw, wind_angle):
    wind_rad = math.radians(wind_angle)
    center_x, center_y = 9, 9

    tip_x = center_x + 7 * math.sin(wind_rad)
    tip_y = center_y - 7 * math.cos(wind_rad)
    tail_x = center_x - 7 * math.sin(wind_rad)
    tail_y = center_y + 7 * math.cos(wind_rad)
    draw.line([(tail_x, tail_y), (tip_x, tip_y)], fill=WIND_COLOR, width=1)

    left_rad = math.radians(wind_angle - 135)
    right_rad = math.radians(wind_angle + 135)

    draw.line(
        [
            (tip_x, tip_y),
            (tip_x + 3 * math.sin(left_rad), tip_y - 3 * math.cos(left_rad))
        ],
        fill=WIND_COLOR,
        width=1
    )

    draw.line(
        [
            (tip_x, tip_y),
            (tip_x + 3 * math.sin(right_rad), tip_y - 3 * math.cos(right_rad))
        ],
        fill=WIND_COLOR,
        width=1
    )


def append_data(base_image, state, filename, map_size=256):
    extended_image = Image.new("RGB", (map_size, map_size + 42), (255, 255, 255))
    extended_image.paste(base_image, (0, 0))
    text_draw = ImageDraw.Draw(extended_image)

    b_x = state['boat_x']
    b_y_screen = map_size - state['boat_y']
    boat_rad = math.radians(state['boat_angle'])
    wind_rad = math.radians(state['wind_angle'])

    u_x = math.sin(boat_rad)
    u_y = -math.cos(boat_rad)

    # Force & Drag Calc
    drive_efficiency = 0.5 + 0.5 * math.cos(math.radians(state['theta_rel'] + 180))
    raw_fwd_force = state['wind_strength'] * drive_efficiency * 0.15

    drag_coefficient = 0.05
    drag = drag_coefficient * (state['boat_speed'] ** 2)
    net_dv = raw_fwd_force - drag

    # Telemetry Output
    line1 = (f"Wind: {state['wind_angle']:.0f} deg | "
             f"Str: {state['wind_strength']:.1f} | "
             f"Spd: {state['boat_speed']:.1f}")
    line2 = (f"Boat: {state['boat_angle']:.0f} deg | "
             f"Sail: {state['sail_angle']:.0f} "
             f"(Opt: {state['theta_opt']:.0f})")
    line3 = (f"Sail Size: {state['sail_size']:.1f} px | "
             f"Rel Wind: {state['theta_rel']:.0f} deg")
    line4 = (f"Fwd Force: {raw_fwd_force:.2f} | "
             f"Drag: {drag:.2f} | "
             f"Net dV: {net_dv:+.2f}")

    text_draw.text((5, 257), line1, fill=(0, 0, 0))
    text_draw.text((5, 267), line2, fill=(0, 0, 0))
    text_draw.text((5, 277), line3, fill=(0, 0, 0))
    text_draw.text((5, 287), line4, fill=(0, 0, 0))

    # Setup: Vector Line Overlay
    anno_image = extended_image.copy()
    anno_draw = ImageDraw.Draw(anno_image)
    force_image = extended_image.copy()
    force_draw = ImageDraw.Draw(force_image)

    # Speed Vector (Blue)
    visual_multiplier = 5
    speed_mag = state['boat_speed'] * visual_multiplier
    dx_speed = speed_mag * u_x
    dy_speed = speed_mag * u_y

    front_x = b_x + 3 * u_x
    front_y = b_y_screen + 3 * u_y
    speed_start_x = front_x + 5 * u_x
    speed_start_y = front_y + 5 * u_y
    anno_draw.line(
        [
            (speed_start_x, speed_start_y),
            (speed_start_x + dx_speed,
             speed_start_y + dy_speed)
        ],
        fill=SPEED_VEC_COLOR,
        width=1)

    # Wind Force Acting on Boat (Green)
    wind_mag = state['wind_strength'] * 1.5
    dx_wind = wind_mag * math.sin(wind_rad)
    dy_wind = -wind_mag * math.cos(wind_rad)

    w_end_x = b_x - 5 * math.sin(wind_rad)
    w_end_y = b_y_screen + 5 * math.cos(wind_rad)
    w_start_x = b_x - (wind_mag + 5) * math.sin(wind_rad)
    w_start_y = b_y_screen + (wind_mag + 5) * math.cos(wind_rad)
    anno_draw.line([(w_start_x, w_start_y), (w_end_x, w_end_y)], fill=WIND_VEC_COLOR, width=1)

    # Optimal Sail Bearing (Yellow)
    pivot_x = b_x + 2 * u_x
    pivot_y = b_y_screen + 2 * u_y
    opt_rad = math.radians(state['boat_angle'] + 180 + state['theta_opt'])
    opt_start_x = pivot_x + 5 * math.sin(opt_rad)
    opt_start_y = pivot_y - 5 * math.cos(opt_rad)
    opt_end_x = opt_start_x + 10 * math.sin(opt_rad)
    opt_end_y = opt_start_y - 10 * math.cos(opt_rad)
    anno_draw.line([(opt_start_x, opt_start_y), (opt_end_x, opt_end_y)], fill=OPT_SAIL_COLOR, width=1)

    # Resultant Force Vectors (Orange & Cyan)
    dx_res = dx_speed + dx_wind
    dy_res = dy_speed + dy_wind
    force_draw.line([(b_x, b_y_screen), (b_x + dx_res, b_y_screen + dy_res)], fill=RESULTANT_COLOR, width=1)

    drawn_fwd_mag = raw_fwd_force * 15
    dx_fwd = drawn_fwd_mag * u_x
    dy_fwd = drawn_fwd_mag * u_y
    force_draw.line([(b_x, b_y_screen), (b_x + dx_fwd, b_y_screen + dy_fwd)], fill=FWD_FORCE_COLOR, width=2)

    # Scaled for visual output and NOT for models use! 1024 x 1024
    scale_factor = 4
    final_width = extended_image.width * scale_factor
    final_height = extended_image.height * scale_factor

    extended_image.resize((final_width, final_height), Image.Resampling.NEAREST).save(f"{filename}.png")
    anno_image.resize((final_width, final_height), Image.Resampling.NEAREST).save(f"{filename}_annotated.png")
    force_image.resize((final_width, final_height), Image.Resampling.NEAREST).save(f"{filename}_force.png")


def generate_valid_environment(map_size=256):
    attempts = 0
    while True:
        attempts += 1
        image = Image.new("RGB", (map_size, map_size), WATER_COLOR)
        draw = ImageDraw.Draw(image)

        num_landmasses = random.randint(2, 11)
        for i in range(num_landmasses):
            base_radius = random.uniform(10.0, 35.0)
            noise_scale = random.uniform(0.7, 1.5)
            amplitude = random.uniform(8.0, 15.0)
            center_x, center_y = random.uniform(0, map_size), random.uniform(0, map_size)

            points = []
            num_steps = 100 
            for step in range(num_steps):
                theta = (step / num_steps) * (2 * math.pi)
                noise_val = pnoise2(math.cos(theta) * noise_scale + center_x, 
                                    math.sin(theta) * noise_scale + center_y, octaves=4)
                r = base_radius + (noise_val * amplitude)
                points.append((center_x + r * math.cos(theta), center_y + r * math.sin(theta)))
            draw.polygon(points, fill=LAND_COLOR)

        pixels = image.load()
        land_pixels = sum(1 for x in range(map_size) for y in range(map_size) if pixels[x, y] == LAND_COLOR)
        if land_pixels / (map_size * map_size) > 0.25: continue  

        # Exclusion box to protect starting point (X: 78 to 178, Y: 176 to 256)
        if any(
            pixels[max(0, min(255, x)), max(0, min(255, y))] == LAND_COLOR 
            for x in range(78, 178)
            for y in range(176, 256)
        ):
            continue

        if any(
            pixels[max(0, min(255, x)), max(0, min(255, y))] == LAND_COLOR 
            for x in range(64, 193)
            for y in range(0, 17)
        ):
            continue

        # Safe Starting Logic (Goal-Oriented: ie. North facing if poss)
        new_wind_angle = random.uniform(0, 360)
        wind_source = (new_wind_angle + 180) % 360
        desired_heading = random.uniform(-90, 90) % 360

        twa_check = (wind_source - desired_heading + 180) % 360 - 180
        no_go_limit = 45.0

        if abs(twa_check) < no_go_limit:
            if twa_check >= 0:
                new_boat_angle = (wind_source - no_go_limit) % 360
            else:
                new_boat_angle = (wind_source + no_go_limit) % 360
        else:
            new_boat_angle = desired_heading

        final_twa = (wind_source - new_boat_angle + 180) % 360 - 180
        new_sail_angle = -(final_twa / 2.0) + random.uniform(-10, 10)

        state = {
            'boat_x': random.randint(120, 136),
            'boat_y': random.randint(15, 30),
            'boat_angle': new_boat_angle,
            'boat_speed': random.uniform(1.0, 5.0),
            'sail_angle': new_sail_angle,
            'sail_size': random.uniform(3.0, 6.0),
            'wind_angle': new_wind_angle,
            'wind_strength': random.uniform(5.0, 20.0),
            'theta_rel': final_twa,
            'theta_opt': -(final_twa / 2.0)
        }

        draw_boat(
            draw,
            state['boat_x'],
            state['boat_y'],
            state['boat_angle'],
            state['sail_angle'],
            state['sail_size'],
            map_size
        )
        draw_wind_arrow(draw, state['wind_angle'])

        return image, state


def main():
    print("Generating environments...")
    for i in range(5):
        valid_map, state = generate_valid_environment()
        append_data(valid_map, state, filename=f"map_{i}")
        print(f"Map {i+1} Output Complete.")


if __name__ == "__main__":
    main()
