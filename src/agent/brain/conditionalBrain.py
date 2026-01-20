import numpy as np
from copy import deepcopy
from agent.brain.brain import Brain
from typing import TYPE_CHECKING, Dict
if TYPE_CHECKING:
    from world.world import World


class ConditionalBrain(Brain):
    
    @classmethod
    def clone(cls, other):
        brain = deepcopy(other)
        brain._init_vars()
        return brain
    
    def __init__(self) -> None:
        self.attack_distance = 1000
        self.follow_min_distance = 20
        self.follow_max_distance = 1000
        self.flee_distance = 1000
        self.energy_to_reproduce_threshold_ratio = 1.1
        self.starve_threshold_ratio = 0.8
        self._init_vars()
        
    def _init_vars(self):
        self.last_action = 'ile'
        self.context = {}
        self.memory_time = 1000
        self.memory_counter = 0
        
    def sense(self, agent, world: 'World'):
        super().sense(agent, world)
        self.context['nearest_enemy_genome_dist'] = world.speciator.genome_distance(agent.genome, self.context['nearest_enemy'].genome) if self.context['nearest_enemy'] is not None else 0
        
    def decide(self):
        if self.context['self_genome'].reproduction == 'mate':
            can_reproduce = self.context['nearest_ally'] is not None and self.context['ready_to_reproduce'] and self.context['nearest_ally'].ready_to_reproduce
        else: # clone
            can_reproduce = self.context['ready_to_reproduce'] # and self.context['self_genome'].energy_to_reproduce * self.energy_to_reproduce_threshold_ratio > self.context['energy']
        can_attack = self.context['nearest_enemy'] is not None
        can_flee = self.context['nearest_enemy'] is not None
        can_follow = self.context['nearest_ally'] is not None
        can_eat = self.context['nearest_food'] is not None 
        # last goal
        if self.memory_counter <=0:
            self.last_goal = None
        if self.last_goal is not None and not self.last_goal.completed:
            self.memory_counter -= 1
            if self.last_action == 'reproduce' and can_reproduce:
                self.action = 'reproduce'
                return 'reproduce'
            if self.last_action == 'attack' and can_attack and not can_reproduce:
                self.action = 'attack'
                return 'attack'
            if self.last_action == 'eat' and can_eat and not can_reproduce:
                self.action = 'eat'
                return 'eat' 
        if self.context['self_genome'].reproduction == 'mate':  
            will_reproduce = can_reproduce and self.context['nearest_agent_genome_dist'] < self.context['self_genome'].partner_genome_distance
        else: # clone
            will_reproduce = can_reproduce
        will_attack = can_attack and np.random.rand() < self.context['self_genome'].aggressive and self.context['nearest_enemy_genome_dist'] > self.context['self_genome'].partner_genome_distance and self.context['nearest_enemy_dist'] < self.attack_distance
        will_flee = can_flee and self.context['nearest_enemy_genome_dist'] > self.context['self_genome'].partner_genome_distance and self.context['nearest_enemy_dist'] < self.flee_distance
        will_follow = can_follow and np.random.rand() < self.context['self_genome'].social and self.follow_min_distance < self.context['nearest_ally_dist'] < self.follow_max_distance
        will_eat = can_eat and self.context["energy"] < self.starve_threshold_ratio
        # mating
        self.action = 'wander'
        if will_reproduce:
            self.action = 'reproduce'
        elif will_attack:
            self.action = 'attack'
        elif will_flee:
            self.action = 'flee'   
        elif will_eat:
            self.action = 'eat'
        elif will_follow:
            self.action = 'follow'
        self.last_action = self.action
        self.memory_counter = self.memory_time
        return self.action
    
    def mutate(self, scale_factor: float = 0, n_mutations: int | None = -1):
        pass
    
    def get_data(self) -> Dict:
        return {'weight' : 0}