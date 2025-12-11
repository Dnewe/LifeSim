import numpy as np
import utils.pos_utils as posUtils
from utils.pos_utils import Pos
from typing import Tuple
from noise import snoise2



class FoodMap():

    def __init__(self, w, h, base_color=[255,0,0], base_energy=100, base_size=3, rate_per_pixel = 1e-5) -> None:
        self.base_color = base_color
        self.base_energy = base_energy
        self.base_size = base_size
        self.rate_per_pixel = rate_per_pixel
        self.width = w
        self.height = h
        self.biome_arr = np.zeros((w, h), dtype=np.float32)
        self.food_arr = np.zeros((w, h), dtype=np.int32)

    def generate_biomes(self, scale= 0.0005, octaves= 2, exponent=3):
        x, y = np.meshgrid(
            np.arange(self.width) * scale,
            np.arange(self.height) * scale,
            indexing='ij'
        )
        noise_grid = np.vectorize(lambda yy, xx: snoise2(xx, yy, octaves))(y, x)
        self.biome_arr = ((noise_grid + 1)*(0.5)).astype(np.float32)
        self.biome_arr = self.biome_arr**exponent


    def step(self, factor=1):
        factor = factor * self.rate_per_pixel
        rand = np.random.random(self.biome_arr.shape)
        spawn = (rand < factor* self.biome_arr)
        energy = np.zeros_like(self.biome_arr, dtype=np.float32)
        np.divide(0.01, self.biome_arr, out=energy, where=self.biome_arr > 0)
        self.food_arr += spawn * self.base_energy # energy.astype(np.int32)


    def add_food(self, pos: Pos, energy):
        self.food_arr[*pos] = energy

    def delete_food(self, pos: Pos):
        self.food_arr[*pos] = 0
    
    def get_food_energy(self, pos: Pos):
        return self.food_arr[*pos]
    
    def get_food_size(self, pos: Pos):
        energy = self.food_arr[*pos]
        size = self.base_size*(1/self.base_energy * energy)**0.5
        return int(size)
    
    def get_food_color(self, pos: Pos):
        energy = self.food_arr[*pos]
        color = self.base_color
        color[1] = (energy*0.01)%255
        return color
    
    def get_nearest_food(self, pos: Pos, radius: int) -> Tuple[Pos|None, float]:
        x, y = pos
        # get food in square radius
        x_min, x_max, y_min, y_max = posUtils.square(pos, radius, self.width, self.height)
        region = self.food_arr[x_min:x_max, y_min:y_max]
        xs, ys = np.where(region!=0)
        if len(xs) == 0:
            return None, 0
        xs += x_min
        ys += y_min
        dx = xs - x
        dy = ys - y
        dist = np.hypot(dx, dy)
        i = np.argmin(dist)
        # check if in circle radius
        if dist[i] > radius:
            return None, 0
        return (xs[i], ys[i]), dist[i]
