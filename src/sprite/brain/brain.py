from copy import deepcopy
import sys
import numpy as np
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Dict, Any, Tuple, Self
if TYPE_CHECKING:
    from sprite.agent import Agent
    from world.world import World
import utils.timeperf as timeperf


class Brain(ABC):
    key_to_idx: dict[Any, int]
    n_keys: int
    mutation_scale: float
    n_mutations: int
    
    @classmethod
    def from_config(cls, config):
        match config['type']:
            case 'conditional':
                from sprite.brain.conditionalBrain import ConditionalBrain
                return ConditionalBrain()
            case 'utility':
                from sprite.brain.utilityFunBrain import UtilityFunBrain
                return UtilityFunBrain.from_config(config)
            case _:
                print(f'brain {type} not implemented')
                sys.exit(1)

    @classmethod
    def clone(cls, other):
        brain = deepcopy(other)
        return brain
                
    @classmethod
    def from_brain1_brain2(cls, brain1, brain2):
        return np.random.choice([brain1, brain2]) # TODO
    
    def get_context(self, agent: 'Agent', world: 'World'):
        near_agents = world.get_near_agents(agent, int(agent.genome.vision_range))
        nearest_food_pos, nearest_food_dist = world.get_nearest_food(agent.get_pos(), int(agent.genome.vision_range))
        nearest_food_energy = world.foodmap.get_food_energy(nearest_food_pos) if nearest_food_pos is not None else 0
        nearest_food_size = world.foodmap.get_food_size(nearest_food_pos) if nearest_food_pos is not None else 0
        return {
            'self_agent': agent,
            'near_agents': near_agents,
            'nearest_food_pos': nearest_food_pos,
            'nearest_food_dist': nearest_food_dist,
            'nearest_food_energy': nearest_food_energy,
            'nearest_food_size': nearest_food_size
        }
    
    def sense(self, agent, world):
        self.context = self.get_context(agent, world)

    @abstractmethod
    def decide(self) -> Tuple[str, Any]:
        ...
    
    @abstractmethod
    def mutate(self, scale_factor:float =1., n_mutations: int|None =-1):
        ...
    
    @classmethod
    @timeperf.timed()
    def distance(cls, brain1: Self, brain2: Self, brains_std, alpha:float=1., eps:float=1e-9) -> float:
        v1 = brain1.to_vector()
        v2 = brain2.to_vector()
        weights = np.array([1.0 for k in cls.key_to_idx]) # TODO weights for actions
        # normalized euclidean
        diff = (v1 - v2) / (brains_std + eps)
        euclid = np.sqrt(np.sum(weights * diff * diff))
        # cosine distance (directional)
        cosine = 1.0 - np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + eps)
        d = euclid * (1.0 + alpha * cosine) 
        d /= cls.n_keys
        return d
    
    @abstractmethod  
    def to_vector(self) -> np.ndarray:
        ...
    
    @abstractmethod
    def get_data(self) -> Dict:
        ...
