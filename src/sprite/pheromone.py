from sprite.sprite import Sprite
import utils.pos_utils as posUtils
from typing import Tuple


class Pheromone(Sprite):
    # attributes
    intensity: float
    radius: float
    #color TODO
    
    def __init__(self, pos: posUtils.Pos, intensity, radius) -> None:
        self.x, self.y = pos
        self.intensity = intensity
        self.radius = radius
        
    def get_color(self, mode: str) -> Tuple[int, int, int]:
        return (255,0,0) # TODO