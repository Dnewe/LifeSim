import numpy as np
import utils.pos_utils as posUtils
from world.world import World


class Frame():
    def __init__(self, w, h, cam_x, cam_y, shm, agent_display_mode='specie') -> None:
        self.agent_display_mode = agent_display_mode
        self.w = w
        self.h = h
        self.cam_x = cam_x
        self.cam_y = cam_y
        self.array = np.ndarray((w, h, 3), dtype=np.uint8, buffer=shm.buf)
        self.array[:] = 0
        # options
        self.show_biomes = True
        self.show_food = True
        self.show_agents = True
        
    def render(self, world: World):
        # window view
        self.x0, self.y0 = self.cam_x.value, self.cam_y.value
        self.x1, self.y1 = self.x0 + self.w, self.y0 + self.h
        # render
        if self.show_biomes: self.render_biome(world)
        if self.show_food: self.render_food(world)
        if self.show_agents: self.render_agents(world)
        
                
    def render_biome(self, world: World):
        visible_biome = world.foodmap.biome_arr[self.x0:self.x1, self.y0:self.y1]
        self.array[:] = world.foodmap.biome_LUT[visible_biome]
        
    def render_food(self, world: World):
        visible_food = world.foodmap.food_arr[self.x0:self.x1, self.y0:self.y1]
        mask = visible_food > 0
        xs, ys = np.where(mask)
        for x, y in zip(xs, ys):
            pos = (x + self.x0, y + self.y0)
            size = world.foodmap.get_food_size(pos)
            x_min, x_max, y_min, y_max = posUtils.square((x,y), size, self.w, self.h)
            self.array[x_min:x_max ,y_min:y_max] = world.foodmap.food_color
            
    def render_agents(self, world: World):
        for agent in world.agents:
            ax, ay = agent.get_pos()
            size = agent.get_size()
            if self.x0 - size <= ax <= self.x1 + size and self.y0 - size <= ay <= self.y1 + size:
                sx = ax - self.x0
                sy = ay - self.y0
                x_min, x_max, y_min, y_max = posUtils.square((sx, sy), size, self.w, self.h)
                self.array[x_min:x_max-1 ,y_min:y_max-1] = agent.get_color(mode=self.agent_display_mode)
                
    def render_pheromone(self, world: World):
        pass