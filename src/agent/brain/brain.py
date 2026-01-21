from copy import deepcopy
import sys
import numpy as np
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Dict
if TYPE_CHECKING:
    from agent.agent import Agent
    from world.world import World


class Brain(ABC):
    action: str = 'idle'
    mutation_scale: float
    n_mutations: int
    
    @classmethod
    def from_config(cls, config):
        match config['type']:
            case 'conditional':
                from agent.brain.conditionalBrain import ConditionalBrain
                return ConditionalBrain()
            case 'utility':
                from agent.brain.utilityFunBrain import UtilityFunBrain
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
        near_enemies = []
        near_allies = []
        for a,d in near_agents:
            if a.species == agent.species:
                near_allies.append((a,d))
            else:
                near_enemies.append((a,d))
        nearest_food, nearest_food_dist = world.get_nearest_food(agent.get_pos(), int(agent.genome.vision_range))
        nearest_food_energy = world.foodmap.get_food_energy(nearest_food) if nearest_food is not None else 0
        return {
            # agent variables
            'energy': agent.energy / agent.genome.max_energy - 0.5,
            'satiety': agent.satiety / agent.genome.max_satiety - 0.5,
            'health': agent.health / agent.genome.max_health - 0.5,
            'age': agent.age / agent.genome.max_age - 0.5,
            'ready_to_reproduce': int(agent.ready_to_reproduce) - 0.5,
            # agent properties
            'self_genome': agent.genome,
            # environment
            'n_allies': len(near_allies) / 10  - 0.5,
            'n_enemies': len(near_enemies) / 10  - 0.5,
            'nearest_ally': near_allies[0][0] if len(near_allies)>0 else None,
            'nearest_ally_dist': near_allies[0][1] / agent.genome.vision_range - 0.5 if len(near_allies)>0 else 0.5, 
            'nearest_enemy': near_enemies[0][0] if len(near_enemies)>0 else None,
            'nearest_enemy_dist': near_enemies[0][1] / agent.genome.vision_range - 0.5 if len(near_enemies)>0 else 0.5, 
            'nearest_food': nearest_food,
            'nearest_food_dist': nearest_food_dist / agent.genome.vision_range - 0.5,
            'nearest_food_energy': nearest_food_energy / world.foodmap.food_base_energy - 0.5,
        }
    
    def sense(self, agent, world):
        self.context = self.get_context(agent, world)
    
    @abstractmethod
    def get_data(self) -> Dict:
        ...

    @abstractmethod
    def decide(self) -> str:
        ...
    
    @abstractmethod
    def mutate(self, scale_factor:float =1., n_mutations: int|None =-1):
        ...
