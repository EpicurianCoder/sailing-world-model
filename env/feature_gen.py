import math
from PIL import Image, ImageDraw
import config
import random
from noise import pnoise2


class FeatureGenerator:
    def gen_land_feature(map_size=config.MAP_SIZE):
        attempts = 0
        while True:
            attempts += 1
            feature_image = Image.new("L", (map_size, map_size), 0)
            draw = ImageDraw.Draw(feature_image)

            # Use perlin noice and circles with aplitutde value to draw islands.
            num_landmasses = random.randint(5, 13)
            for i in range(num_landmasses):
                base_radius = random.uniform(14.0, 35.0)
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

                draw.polygon(points, fill=1)

            pixels = feature_image.load()
            land_pixels = sum(1 for x in range(map_size) for y in range(map_size) if pixels[x, y] == 1)
            coverage = land_pixels / (map_size * map_size)

            # Dont Cover more than 25%
            if coverage > 0.25:
                continue
            # Protect the boat staring box
            if any(pixels[max(0, min(map_size-1, x)), max(0, min(map_size-1, y))] == 1 for x in range(78, 178) for y in range(176, 256)):
                continue
            # Dont cover the fininsh line
            if any(pixels[max(0, min(map_size-1, x)), max(0, min(map_size-1, y))] == 1 for x in range(64, 193) for y in range(0, 17)):
                continue

            return feature_image, num_landmasses, coverage

    def get_boat_feature_layer(x, y, boat_angle, map_size=config.MAP_SIZE):
        # "L" = single-channel 8-bit image
        feature_image = Image.new("L", (map_size, map_size), 0)
        draw = ImageDraw.Draw(feature_image)

        screen_y = map_size - y
        boat_rad = math.radians(boat_angle)
        front_x = x + 5 * math.sin(boat_rad)
        front_y = screen_y - 5 * math.cos(boat_rad)
        back_x = x - 5 * math.sin(boat_rad)
        back_y = screen_y + 5 * math.cos(boat_rad)
        draw.line([(back_x, back_y), (front_x, front_y)], fill=1, width=3)

        return feature_image

    def get_sail_feature_layer(x, y, boat_angle, sail_angle, sail_size, map_size=config.MAP_SIZE):
        feature_image = Image.new("L", (map_size, map_size), 0)
        draw = ImageDraw.Draw(feature_image)

        screen_y = map_size - y
        boat_rad = math.radians(boat_angle)
        sail_rad = math.radians(boat_angle + sail_angle + 180)

        # The Pivot Point (nightmare!!)
        pivot_x = x + 2 * math.sin(boat_rad)
        pivot_y = screen_y - 2 * math.cos(boat_rad)
        sail_end_x = pivot_x + sail_size * math.sin(sail_rad)
        sail_end_y = pivot_y - sail_size * math.cos(sail_rad)

        draw.line([(pivot_x, pivot_y), (sail_end_x, sail_end_y)], fill=1, width=1)

        return feature_image
