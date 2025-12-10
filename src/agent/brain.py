from agent.goal import *
import sys
from typing import TYPE_CHECKING, Dict
if TYPE_CHECKING:
    from agent.agent import Agent
import numpy as np


DEFAULT = 'conditional'


class Brain():
    config: Dict
    
    @classmethod
    def from_config(cls, config):
        match config['type']:
            case 'conditional':
                return ConditionalBrain()
            case _:
                print(f'brain {type} not implemented')
                sys.exit(1)
                
    @classmethod
    def from_brain1_brain2(cls, brain1, brain2):
        return np.random.choice([brain1, brain2]) # TODO
    
    def sense(self, agent, world):
        pass

    def decide(self) -> Goal:
        return IdleGoal()
    
    def unpack_context(self, context):
        for k, v in context.items():
            setattr(self, k, v)
    
    def update(self, death_reason):
        pass
    

class ConditionalBrain(Brain):
    # TODO has the variables of flee_distance, follow_distances, etcc
    def __init__(self) -> None:
        self.type = 'conditional'
        self.flee_distance = 5
        self.starve_threshold_ratio = 0.8
        self.context = {}
        
    def sense(self, agent: 'Agent', world: 'World'):
        near_agent, near_agent_dist = world.get_nearest_agent(agent, int(agent.dna.vision_range))
        near_food, near_food_dist = world.get_nearest_food(agent.get_pos(), int(agent.dna.vision_range))
        self.context = {
            'self_dna': agent.dna,
            'energy': agent.energy,
            'ready_to_mate': agent.ready_to_mate,
            'near_agent': near_agent,
            'near_agent_dist': near_agent_dist,
            'near_food': near_food,
            'near_food_dist': near_food_dist,
        }
        
    def decide(self):
        # mating
        if self.context['near_agent'] is not None and self.context['ready_to_mate'] and self.context['near_agent'].ready_to_mate:
            return MateGoal(partner = self.context['near_agent'], partner_dist = self.context['near_agent_dist'])
        # attacking
        if self.context['near_agent'] is not None and False:
            return AttackGoal(target = self.context['near_agent'], target_dist = self.context['near_agent_dist'])
        # fleeing
        if self.context['near_agent'] is not None and self.context['near_agent_dist'] < self.flee_distance:
            return FleeGoal(pos_to_flee = self.context['near_agent'].get_pos())     
        # eating
        if self.context['near_food'] is not None and self.context["energy"] < self.context["self_dna"].max_energy * self.starve_threshold_ratio:
            return EatGoal(food_pos = self.context['near_food'], food_dist = self.context['near_food_dist']) 
        # wander
        return WanderGoal()