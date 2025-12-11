import numpy as np
import utils.pos_utils as posUtils
from world import World


class Frame():
    def __init__(self, w, h, cam_x, cam_y, shm, agent_display_mode='specie') -> None:
        self.agent_display_mode = agent_display_mode
        self.w = w
        self.h = h
        self.cam_x = cam_x
        self.cam_y = cam_y
        self.array = np.ndarray((w, h, 3), dtype=np.uint8, buffer=shm.buf)
        self.array[:] = 0
        
    def draw(self, world: World):
        # window view
        x0, y0 = self.cam_x.value, self.cam_y.value
        x1, y1 = x0 + self.w, y0 + self.h
        
        # biomes
        visible_biome = world.food_map.biome_arr[x0:x1, y0:y1]
        factor = 100 / world.food_map.biome_arr.max()
        self.array[:] = visible_biome[..., None] * factor
        
        # foods
        visible_food = world.food_map.food_arr[x0:x1, y0:y1]
        mask = visible_food > 0
        xs, ys = np.where(mask)
        for window_pos in zip(xs, ys):
            pos = (window_pos[0] + x0, window_pos[1] + y0)
            size = world.food_map.get_food_size(pos)
            x_min, x_max, y_min, y_max = posUtils.square(window_pos, size, self.w, self.h)
            self.array[x_min:x_max ,y_min:y_max] = world.food_map.get_food_color(pos)
            # TODO optimize food for food size = 1
        
        # agents
        for agent in world.agents:
            ax, ay = agent.get_pos()
            size = agent.get_size()
            if x0 - size <= ax <= x1 + size and y0 - size <= ay <= y1 + size:
                sx = ax - x0
                sy = ay - y0
                x_min, x_max, y_min, y_max = posUtils.square((sx, sy), size, self.w, self.h)
                self.array[x_min:x_max ,y_min:y_max] = agent.get_color(mode=self.agent_display_mode)
