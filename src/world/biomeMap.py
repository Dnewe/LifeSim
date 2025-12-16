from typing import Tuple
import numpy as np
from utils import pos_utils as posUtils
from noise import snoise2


class BiomeMap():
    
    @classmethod
    def from_configs(cls, worldgen_config, food_config):
        return cls(worldgen_config['width'], 
                   worldgen_config['height'], 
                   worldgen_config['noise_map']['scale'], 
                   worldgen_config['noise_map']['octaves'], 
                   worldgen_config['biomes'],
                   food_config['spawn_rate'],
                   food_config['energy'],
                   food_config['size'],
                   food_config['color'])
        
    def __init__(self, w, h, scale, octaves, biomes_config, 
                 food_base_spawn_rate, food_base_energy, food_base_size, food_base_color) -> None:
        # params
        self.biomes_params = self._get_biome_params(biomes_config, food_base_spawn_rate, food_base_energy)
        self.n_biomes = len(self.biomes_params)
        self.food_size_factor = (food_base_size / food_base_energy )**0.5
        self.food_color = food_base_color
        # arrays
        self.biome_arr = np.zeros((w,h)).astype(np.uint8)
        self.food_arr = np.zeros((w,h)).astype(np.uint16)
        self.biome_LUT = self._get_biome_LUT()
        self.generate_biomes(w, h, scale, octaves)
        
    def _get_biome_params(self, biome_config, food_spawn_rate, food_base_energy):
        biome_params = {}
        biome_config_ordered = {b: v for b,v in sorted(biome_config.items(), key= lambda item: item[1]['spawn_order'])}
        acc = 0.
        for i, (_, p) in enumerate(biome_config_ordered.items()):
            params = {
                'spawn_thresh': acc + p['spawn_rate'],
                'food_spawn_rate': food_spawn_rate * p['food_spawn_rate_factor'],
                'food_value': food_base_energy * p['food_energy_factor'],
                'color': p['color']
            }
            biome_params[i] = params
            acc += p['spawn_rate']
        return biome_params
    
    def _get_biome_LUT(self):
        biome_LUT = np.zeros((self.n_biomes, 3), dtype=np.uint8)
        for i, params in self.biomes_params.items():
            biome_LUT[i] = params['color']
        return biome_LUT
            
    def generate_biomes(self, w, h, scale= 0.001, octaves= 2):
        x, y = np.meshgrid(np.arange(w) * scale, np.arange(h) * scale, indexing='ij')
        noise_grid = np.vectorize(lambda yy, xx: snoise2(xx, yy, octaves))(y, x)
        noise_grid = ((noise_grid + 1)*(0.5)).astype(np.float32)
        for k, rules in reversed(list(self.biomes_params.items())):   # biomes in reverse order of spawn order
            self.biome_arr[noise_grid < rules['spawn_thresh']] = k
            
    def spawn_food(self, n_spawns=1):
        for biome_id, params in self.biomes_params.items():
            mask = (self.biome_arr == biome_id)
            spawn = np.random.random(mask.shape) < params["food_spawn_rate"] * n_spawns
            self.food_arr[mask & spawn] = params["food_value"]
            
    def add_food(self, pos, energy):
        self.food_arr[*pos] = energy
        
    def del_food(self, pos):
        self.food_arr[*pos] = 0
            
    def get_food_energy(self, pos):
        return float(self.food_arr[*pos])
    
    def get_food_size(self, pos):
        return int(self.food_size_factor * self.food_arr[*pos]**0.5)
    
    def get_nearest_food(self, pos, radius) -> Tuple[posUtils.Pos|None, float]:
        x, y = pos
        xmin, xmax, ymin, ymax = posUtils.square(pos, radius)
        region = self.food_arr[xmin:xmax, ymin:ymax]
        xs, ys = np.where(region > 0)
        if len(xs) == 0:
            return None, 0
        xs += xmin
        ys += ymin
        dx = xs - x
        dy = ys - y
        dist = np.hypot(dx, dy)
        i = np.argmin(dist)
        # check if in circle radius
        if dist[i] > radius:
            return None, 0
        return (xs[i], ys[i]), dist[i]
        
    

    
    
        