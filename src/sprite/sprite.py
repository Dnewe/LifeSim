import utils.pos_utils as posUtils
from abc import ABC, abstractmethod
from typing import Tuple


class Sprite(ABC):
    # pos
    x:float
    y:float
    cell_x:int
    cell_y:int
    
    def get_pos(self):
        return posUtils.clip_pos((int(self.x), int(self.y)))
    
    @abstractmethod
    def get_color(self, mode:str) -> Tuple[int,int,int]:
        ...