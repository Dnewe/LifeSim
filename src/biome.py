import numpy as np
from noise import snoise2


class Biome():
    @classmethod
    def from_config(cls, config):
        octaves = config['octave']
        exponent = config['exponent']
        
    def __init__(self, w, h, octaves, exponent) -> None:
        self.w = w
        self.h = h
        self.octaves = octaves
        self.exponent = exponent
    
    def generate_biomes(self, scale= 0.001, octaves= 2, exponent=2):
        x, y = np.meshgrid(
            np.arange(self.w) * scale,
            np.arange(self.h) * scale,
            indexing='ij'
        )
        noise_grid = np.vectorize(lambda yy, xx: snoise2(xx, yy, octaves))(y, x)
        self.biome_arr = ((noise_grid + 1)*(0.5)).astype(np.float32)
        self.biome_arr = self.biome_arr**exponent