from sprite.sprite import Sprite
import utils.pos_utils as posUtils
from typing import Tuple


class Pheromone(Sprite):
    # attributes
    intensity: float
    radius: float
    #color TODO
    
    def __init__(self, pos: posUtils.Pos, radius, intensity, decay_rate) -> None:
        self.x, self.y = pos
        self.intensity = intensity
        self.radius = radius
        self.decay_rate = decay_rate
        self.expired = False
        
    def step(self):
        '''
        Decay intensity, check for expired.
        '''
        self.intensity -= self.decay_rate
        if self.intensity <= 0:
            self.expired = True
        
    def get_color(self, mode: str= "default") -> Tuple[int, int, int]:
        return (255,0,0) # TODO