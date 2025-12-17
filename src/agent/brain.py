from agent.goal import *
import sys
from typing import TYPE_CHECKING, Dict
if TYPE_CHECKING:
    from agent.agent import Agent
import numpy as np
from copy import deepcopy


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
    def clone(cls, other):
        brain = deepcopy(other)
        brain._init_vars()
        return brain
                
    @classmethod
    def from_brain1_brain2(cls, brain1, brain2):
        return np.random.choice([brain1, brain2]) # TODO
    
    def _init_vars(self):
        pass
    
    def sense(self, agent, world):
        pass

    def decide(self) -> Goal:
        return IdleGoal()
    
    def unpack_context(self, context):
        for k, v in context.items():
            setattr(self, k, v)
            

class NetworkBrain(Brain):
    def __init__(self) -> None:
        super().__init__()
        
    class Model():
        def __init__(self) -> None:
            pass
        
    

class ConditionalBrain(Brain):
    def __init__(self) -> None:
        self.type = 'conditional'
        self.attack_distance = 50
        self.follow_min_distance = 20
        self.follow_max_distance = 50
        self.flee_distance = 50
        self.energy_to_reproduce_threshold_ratio = 1.1
        self.starve_threshold_ratio = 0.8
        self._init_vars()
        
    def _init_vars(self):
        self.last_goal = None
        self.context = {}
        self.memory_time = 1000
        self.memory_counter = 0
        
    def sense(self, agent: 'Agent', world: 'World'):
        near_agent, near_agent_dist = world.get_nearest_agent(agent, int(agent.dna.vision_range))
        near_food, near_food_dist = world.get_nearest_food(agent.get_pos(), int(agent.dna.vision_range))
        near_agent_dna_dist = world.gc.dna_distance(agent.dna, near_agent.dna) if near_agent is not None else None
        self.context = {
            'self_dna': agent.dna,
            'energy': agent.energy,
            'ready_to_reproduce': agent.ready_to_reproduce,
            'near_agent': near_agent,
            'near_agent_dist': near_agent_dist,
            'near_agent_dna_dist': near_agent_dna_dist,
            'near_food': near_food,
            'near_food_dist': near_food_dist,
        }
        
    def decide(self):
        if self.context['self_dna'].reproduction == 'mate':
            can_reproduce = self.context['near_agent'] is not None and self.context['ready_to_reproduce'] and self.context['near_agent'].ready_to_reproduce
        else: # clone
            can_reproduce = self.context['ready_to_reproduce'] # and self.context['self_dna'].energy_to_reproduce * self.energy_to_reproduce_threshold_ratio > self.context['energy']
        can_attack = self.context['near_agent'] is not None and self.context["energy"] 
        can_flee = self.context['near_agent'] is not None
        can_follow = self.context['near_agent'] is not None
        can_eat = self.context['near_food'] is not None 
        # last goal
        if self.memory_counter <=0:
            self.last_goal = None
        if self.last_goal is not None and not self.last_goal.completed:
            self.memory_counter -= 1
            if isinstance(self.last_goal, ReproduceGoal) and can_reproduce:
                return ReproduceGoal(partner = self.context['near_agent'], partner_dist = self.context['near_agent_dist'])
            if isinstance(self.last_goal, AttackGoal) and can_attack and not can_reproduce:
                return AttackGoal(target = self.context['near_agent'], target_dist = self.context['near_agent_dist'])
            if isinstance(self.last_goal, EatGoal) and can_eat and not can_reproduce:
                return EatGoal(food_pos = self.context['near_food'], food_dist = self.context['near_food_dist']) 
        if self.context['self_dna'].reproduction == 'mate':  
            will_reproduce = can_reproduce and self.context['near_agent_dna_dist'] < self.context['self_dna'].partner_dna_distance
        else: # clone
            will_reproduce = can_reproduce
        will_attack = can_attack and np.random.rand() < self.context['self_dna'].aggressive and self.context['near_agent_dna_dist'] > self.context['self_dna'].partner_dna_distance and self.context['near_agent_dist'] < self.attack_distance
        will_flee = can_flee and self.context['near_agent_dna_dist'] > self.context['self_dna'].partner_dna_distance and self.context['near_agent_dist'] < self.flee_distance
        will_follow = can_follow and np.random.rand() < self.context['self_dna'].social and self.follow_min_distance < self.context['near_agent_dist'] < self.follow_max_distance
        will_eat = can_eat and self.context["energy"] < self.context["self_dna"].max_energy * self.starve_threshold_ratio
        # mating
        goal = WanderGoal() 
        if will_reproduce:
            goal = ReproduceGoal(partner = self.context['near_agent'], partner_dist = self.context['near_agent_dist'])
        elif will_attack:
            goal = AttackGoal(target = self.context['near_agent'], target_dist = self.context['near_agent_dist'])
        elif will_flee:
            goal = FleeGoal(pos_to_flee = self.context['near_agent'].get_pos())     
        elif will_eat:
            goal = EatGoal(food_pos = self.context['near_food'], food_dist = self.context['near_food_dist']) 
        elif will_follow:
            goal = FollowGoal(partner = self.context['near_agent'])
        self.last_goal = goal
        self.memory_counter = self.memory_time
        return goal