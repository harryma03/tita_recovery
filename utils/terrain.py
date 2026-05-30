import numpy as np
from numpy.random import choice

from isaacgym import terrain_utils
from configs import LeggedRobotCfg

class Terrain:
    def __init__(self, cfg: LeggedRobotCfg.terrain, num_robots) -> None:

        self.cfg = cfg
        self.num_robots = num_robots
        self.type = cfg.mesh_type
        if self.type in ["none", 'plane']:
            return
        self.env_length = cfg.terrain_length
        self.env_width = cfg.terrain_width
        self.proportions = [np.sum(cfg.terrain_proportions[:i+1]) for i in range(len(cfg.terrain_proportions))]

        self.cfg.num_sub_terrains = cfg.num_rows * cfg.num_cols
        self.env_origins = np.zeros((cfg.num_rows, cfg.num_cols, 3))

        self.width_per_env_pixels = int(self.env_width / cfg.horizontal_scale)
        self.length_per_env_pixels = int(self.env_length / cfg.horizontal_scale)

        self.border = int(cfg.border_size / self.cfg.horizontal_scale)
        self.tot_cols = int(cfg.num_cols * self.width_per_env_pixels) + 2 * self.border
        self.tot_rows = int(cfg.num_rows * self.length_per_env_pixels) + 2 * self.border

        self.height_field_raw = np.zeros((self.tot_rows, self.tot_cols), dtype=np.int16)
        if cfg.curriculum:
            self.curiculum()
        elif cfg.selected:
            self.selected_terrain()
        else:
            self.randomized_terrain()

        self.heightsamples = self.height_field_raw
        if self.type == "trimesh":
            self.vertices, self.triangles = terrain_utils.convert_heightfield_to_trimesh(self.height_field_raw,
                                                                                         self.cfg.horizontal_scale,
                                                                                         self.cfg.vertical_scale,
                                                                                         self.cfg.slope_treshold)

    def randomized_terrain(self):
        for k in range(self.cfg.num_sub_terrains):
            # Env coordinates in the world
            (i, j) = np.unravel_index(k, (self.cfg.num_rows, self.cfg.num_cols))

            choice = np.random.uniform(0, 1)
            difficulty = np.random.choice([0.5, 0.75, 0.9])
            terrain = self.make_terrain(choice, difficulty)
            self.add_terrain_to_map(terrain, i, j)

    def curiculum(self):
        for j in range(self.cfg.num_cols):
            for i in range(self.cfg.num_rows):
                difficulty = i / self.cfg.num_rows
                choice = j / self.cfg.num_cols + 0.001

                terrain = self.make_terrain(choice, difficulty)
                self.add_terrain_to_map(terrain, i, j)

    def selected_terrain(self):
        terrain_type = self.cfg.terrain_kwargs.pop('type')
        for k in range(self.cfg.num_sub_terrains):
            # Env coordinates in the world
            (i, j) = np.unravel_index(k, (self.cfg.num_rows, self.cfg.num_cols))

            terrain = terrain_utils.SubTerrain("terrain",
                                                width=self.width_per_env_pixels,
                                                length=self.width_per_env_pixels,
                                                vertical_scale=self.vertical_scale,
                                                horizontal_scale=self.horizontal_scale)

            eval(terrain_type)(terrain, **self.cfg.terrain_kwargs.terrain_kwargs)
            self.add_terrain_to_map(terrain, i, j)

    def make_terrain(self, choice, difficulty):
        terrain = terrain_utils.SubTerrain("terrain",
                                            width=self.width_per_env_pixels,
                                            length=self.width_per_env_pixels,
                                            vertical_scale=self.cfg.vertical_scale,
                                            horizontal_scale=self.cfg.horizontal_scale)
        slope = difficulty * 0.4
        step_height = 0.05 + 0.18 * difficulty
        discrete_obstacles_height = 0.05 + difficulty * 0.2
        high_platform_min_height = getattr(self.cfg, "high_platform_min_height", 0.18)
        high_platform_max_height = getattr(self.cfg, "high_platform_max_height", 0.43)
        high_platform_height = high_platform_min_height + (high_platform_max_height - high_platform_min_height) * difficulty
        stepping_stones_size = 1.5 * (1.05 - difficulty)
        stone_distance = 0.05 if difficulty == 0 else 0.1
        gap_size = 1. * difficulty
        pit_depth = 1. * difficulty
        if choice < self.proportions[0]:
            if choice < self.proportions[0] / 2:
                slope *= -1
            terrain_utils.pyramid_sloped_terrain(terrain, slope=slope, platform_size=3.)
        elif choice < self.proportions[1]:
            terrain_utils.pyramid_sloped_terrain(terrain, slope=slope, platform_size=3.)
            terrain_utils.random_uniform_terrain(terrain, min_height=-0.05, max_height=0.05, step=0.005,
                                                    downsampled_scale=0.2)
        elif choice < self.proportions[3]:
            if choice < self.proportions[2]:
                step_height *= -1
            terrain_utils.pyramid_stairs_terrain(terrain, step_width=0.31, step_height=step_height,
                                                    platform_size=3.)
        elif choice < self.proportions[4]:
            num_rectangles = 20
            rectangle_min_size = 1.
            rectangle_max_size = 2.
            terrain_utils.discrete_obstacles_terrain(terrain, discrete_obstacles_height, rectangle_min_size,
                                                        rectangle_max_size, num_rectangles, platform_size=3.)
        elif len(self.proportions) > 5 and choice < self.proportions[5]:
            high_platform_terrain(
                terrain,
                height=high_platform_height,
                approach_length=getattr(self.cfg, "high_platform_approach_length", 0.65),
                platform_length=getattr(self.cfg, "high_platform_length", 2.5),
            )
        elif len(self.proportions) > 6 and choice < self.proportions[6]:
            terrain_utils.stepping_stones_terrain(terrain, stone_size=stepping_stones_size,
                                                    stone_distance=stone_distance, max_height=0., platform_size=4.)
        elif len(self.proportions) > 7 and choice < self.proportions[7]:
            gap_terrain(terrain, gap_size=gap_size, platform_size=3.)
        else:
            pit_terrain(terrain, depth=pit_depth, platform_size=4.)

        return terrain

    def add_terrain_to_map(self, terrain, row, col):
        i = row
        j = col
        # map coordinate system
        start_x = self.border + i * self.length_per_env_pixels
        end_x = self.border + (i + 1) * self.length_per_env_pixels
        start_y = self.border + j * self.width_per_env_pixels
        end_y = self.border + (j + 1) * self.width_per_env_pixels
        self.height_field_raw[start_x: end_x, start_y:end_y] = terrain.height_field_raw

        env_origin_x = (i + 0.5) * self.env_length
        env_origin_y = (j + 0.5) * self.env_width
        x1 = int((self.env_length / 2. - 1) / terrain.horizontal_scale)
        x2 = int((self.env_length / 2. + 1) / terrain.horizontal_scale)
        y1 = int((self.env_width / 2. - 1) / terrain.horizontal_scale)
        y2 = int((self.env_width / 2. + 1) / terrain.horizontal_scale)
        env_origin_z = np.max(terrain.height_field_raw[x1:x2, y1:y2]) * terrain.vertical_scale
        self.env_origins[i, j] = [env_origin_x, env_origin_y, env_origin_z]


def gap_terrain(terrain, gap_size, platform_size=1.):
    gap_size = int(gap_size / terrain.horizontal_scale)
    platform_size = int(platform_size / terrain.horizontal_scale)

    center_x = terrain.length // 2
    center_y = terrain.width // 2
    x1 = (terrain.length - platform_size) // 2
    x2 = x1 + gap_size
    y1 = (terrain.width - platform_size) // 2
    y2 = y1 + gap_size

    terrain.height_field_raw[center_x - x2: center_x + x2, center_y - y2: center_y + y2] = -1000
    terrain.height_field_raw[center_x - x1: center_x + x1, center_y - y1: center_y + y1] = 0


def high_platform_terrain(terrain, height, approach_length=0.65, platform_length=2.5):
    height = int(height / terrain.vertical_scale)
    approach_length = int(approach_length / terrain.horizontal_scale)
    platform_length = int(platform_length / terrain.horizontal_scale)

    start_x = terrain.length // 2 + approach_length
    end_x = min(terrain.length, start_x + platform_length)
    terrain.height_field_raw[start_x:end_x, :] = height


def pit_terrain(terrain, depth, platform_size=1.):
    depth = int(depth / terrain.vertical_scale)
    platform_size = int(platform_size / terrain.horizontal_scale / 2)
    x1 = terrain.length // 2 - platform_size
    x2 = terrain.length // 2 + platform_size
    y1 = terrain.width // 2 - platform_size
    y2 = terrain.width // 2 + platform_size
    terrain.height_field_raw[x1:x2, y1:y2] = -depth

