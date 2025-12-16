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
    

class ConditionalBrain(Brain):
    def __init__(self) -> None:
        self.last_goal = None
        self.type = 'conditional'
        self.attack_distance = 50
        self.follow_min_distance = 20
        self.follow_max_distance = 50
        self.flee_distance = 50
        self.energy_to_mate_threshold_ratio = 1.1
        self.starve_threshold_ratio = 0.8
        self.context = {}
        self.memory_time = 100000 # TODO
        self.memory_counter = 0
        
    def sense(self, agent: 'Agent', world: 'World'):
        near_agent, near_agent_dist = world.get_nearest_agent(agent, int(agent.dna.vision_range))
        near_food, near_food_dist = world.get_nearest_food(agent.get_pos(), int(agent.dna.vision_range))
        near_agent_dna_dist = world.genetic_context.dna_distance(agent.dna, near_agent.dna) if near_agent is not None else None
        self.context = {
            'self_dna': agent.dna,
            'energy': agent.energy,
            'ready_to_mate': agent.ready_to_mate,
            'near_agent': near_agent,
            'near_agent_dist': near_agent_dist,
            'near_agent_dna_dist': near_agent_dna_dist,
            'near_food': near_food,
            'near_food_dist': near_food_dist,
        }
        
    def decide(self):
        can_mate = self.context['near_agent'] is not None and self.context['ready_to_mate'] and self.context['near_agent'].ready_to_mate
        can_attack = self.context['near_agent'] is not None and self.context["energy"] 
        can_flee = self.context['near_agent'] is not None
        can_follow = self.context['near_agent'] is not None
        can_eat = self.context['near_food'] is not None 
        # last goal
        if self.memory_counter <=0:
            self.last_goal = None
        if self.last_goal is not None and not self.last_goal.completed:
            self.memory_counter -= 1
            if isinstance(self.last_goal, MateGoal) and can_mate:
                return MateGoal(partner = self.context['near_agent'], partner_dist = self.context['near_agent_dist'])
            if isinstance(self.last_goal, AttackGoal) and can_attack and not can_mate:
                return AttackGoal(target = self.context['near_agent'], target_dist = self.context['near_agent_dist'])
            if isinstance(self.last_goal, EatGoal) and can_eat and not can_mate:
                return EatGoal(food_pos = self.context['near_food'], food_dist = self.context['near_food_dist']) 
        will_mate = can_mate and self.context['near_agent_dna_dist'] < self.context['self_dna'].partner_dna_distance
        will_attack = can_attack and np.random.rand() < self.context['self_dna'].aggressive and self.context['near_agent_dna_dist'] > self.context['self_dna'].partner_dna_distance and self.context['near_agent_dist'] < self.attack_distance
        will_flee = can_flee and self.context['near_agent_dna_dist'] > self.context['self_dna'].partner_dna_distance and self.context['near_agent_dist'] < self.flee_distance
        will_follow = can_follow and np.random.rand() < self.context['self_dna'].social and self.follow_min_distance < self.context['near_agent_dist'] < self.follow_max_distance
        will_eat = can_eat and self.context["energy"] < self.context["self_dna"].max_energy * self.starve_threshold_ratio
        # mating
        goal = WanderGoal() 
        if will_mate:
            goal = MateGoal(partner = self.context['near_agent'], partner_dist = self.context['near_agent_dist'])
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