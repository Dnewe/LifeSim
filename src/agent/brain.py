from agent.goal import *
import sys
from typing import TYPE_CHECKING, Dict, List
if TYPE_CHECKING:
    from agent.agent import Agent
import numpy as np
from copy import deepcopy


DEFAULT = 'conditional'


###################
###  MetaClass  ###
###################

class Brain():
    config: Dict
    action: str = 'idle'
    
    @classmethod
    def from_config(cls, config):
        match config['type']:
            case 'conditional':
                return ConditionalBrain()
            case 'utility':
                return UtilityBasedBrain.from_config(config)
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
    
    def get_data(self):
        return {'action': {'weight': 0.}}
    
    def get_context(self, agent: 'Agent', world: 'World'):
        near_agents = world.get_near_agents(agent, int(agent.dna.vision_range))
        near_enemies = []
        near_allies = []
        for a,d in near_agents:
            if a.species == agent.species:
                near_allies.append((a,d))
            else:
                near_enemies.append((a,d))
        nearest_food, nearest_food_dist = world.get_nearest_food(agent.get_pos(), int(agent.dna.vision_range))
        nearest_food_energy = world.foodmap.get_food_energy(nearest_food) if nearest_food is not None else 0
        return {
            # agent variables
            'energy': agent.energy / agent.dna.max_energy,
            'satiety': agent.satiety / agent.dna.max_satiety,
            'health': agent.health / agent.dna.max_health,
            'age': agent.age / agent.dna.max_age,
            'ready_to_reproduce': int(agent.ready_to_reproduce),
            # agent properties
            'self_dna': agent.dna,
            # environment
            'n_allies': len(near_allies) / 10,
            'n_enemies': len(near_enemies) / 10,
            'nearest_ally': near_allies[0][0] if len(near_allies)>0 else None,
            'nearest_ally_dist': near_allies[0][1] / agent.dna.vision_range if len(near_allies)>0 else 1., 
            'nearest_enemy': near_enemies[0][0] if len(near_enemies)>0 else None,
            'nearest_enemy_dist': near_enemies[0][1] / agent.dna.vision_range if len(near_enemies)>0 else 1., 
            'nearest_food': nearest_food,
            'nearest_food_dist': nearest_food_dist / agent.dna.vision_range,
            'nearest_food_energy': nearest_food_energy / world.foodmap.food_base_energy,
        }
    
    def sense(self, agent, world):
        self.context = self.get_context(agent, world)

    def decide(self) -> Goal:
        return GOALS_MAP[self.action]()
    
    def mutate(self, scale=0):
        pass
            

###################
###  NeuralNet  ###
###################         

class NetworkBrain(Brain):
    def __init__(self) -> None:
        super().__init__()
        
    class Model():
        def __init__(self) -> None:
            pass


####################
###  UtilityFun  ###
####################
MIN_G = 0.01
MAX_G = 0.9    

MIN_W = -5
MAX_W = +5 

class UtilityScore:
    
    def __init__(self, action:str, context_weights:Dict[str,float]) -> None:
        self.action = action
        self.context_weights = context_weights
        self.prev_u = 0
    
    def compute(self, context:Dict):
        u = 0
        for k,v in context.items():
            if isinstance(v, (int, float)):
                w = max(min(self.context_weights.get(k, 0), MAX_G), MIN_W)
                u += w * v
        u += self.context_weights.get('bias', 0)
        g = max(min(self.context_weights.get('gamma', 0), MAX_G), MIN_G)
        self.context_weights['gamma'] = g
        u = g * self.prev_u + (1-g) * u 
        self.prev_u = u
        return u
    
    def randomize(self, n_weights=None, scale=0.1):
        # TODO update according to heuristics 
        weights_to_change = np.random.choice(list(self.context_weights), n_weights) if n_weights is not None else self.context_weights.keys()
        for k in weights_to_change:
            self.context_weights[k] += (np.random.rand() - 0.5) * (MAX_W - MIN_G) * scale
            

class UtilityBasedBrain(Brain):
    def __init__(self, utility_scores: List[UtilityScore]) -> None:
        self.type = 'utility'
        self.utility_scores = utility_scores
        
    @classmethod
    def from_config(cls, config):
        utility_scores = []
        for k, v in config['utility_scores'].items():
            utility_scores.append(UtilityScore(k, v))
        return cls(utility_scores)        
    
    def decide(self):
        actions_and_scores = [(f.action, f.compute(self.context)) for f in self.utility_scores]
        sorted_actions = [x[0] for x in sorted(actions_and_scores, key= lambda x: x[1], reverse=True)]
        goal = GOALS_MAP[sorted_actions[0]](self.context)
        while not goal.is_valid():
            sorted_actions.pop(0)
            goal = GOALS_MAP[sorted_actions[0]](self.context)
        self.action = sorted_actions[0]
        return goal

    def update(self, reward):
        for us in self.utility_scores:
            us.randomize(None, 0.5)
            
    def mutate(self, scale = 0.1, n_weights=None):
        for us in self.utility_scores:
            us.randomize(n_weights, scale)
            
    def get_data(self):
        data = {}
        for us in self.utility_scores:
            data[us.action] = us.context_weights
        return data
    
    

###################
###    Rules    ###
###################        

class ConditionalBrain(Brain):
    def __init__(self) -> None:
        self.type = 'conditional'
        self.attack_distance = 1000
        self.follow_min_distance = 20
        self.follow_max_distance = 1000
        self.flee_distance = 1000
        self.energy_to_reproduce_threshold_ratio = 1.1
        self.starve_threshold_ratio = 0.8
        self._init_vars()
        
    def _init_vars(self):
        self.last_goal = None
        self.context = {}
        self.memory_time = 1000
        self.memory_counter = 0
        
    def sense(self, agent, world: 'World'):
        super().sense(agent, world)
        self.context['nearest_enemy_dna_dist'] = world.gc.dna_distance(agent.dna, self.context['nearest_enemy'].dna) if self.context['nearest_enemy'] is not None else 0
        
    #def sense(self, agent: 'Agent', world: 'World'):
        '''near_agents_dists = world.get_near_agents(agent, int(agent.dna.vision_range))
        near_agent, near_agent_dist = near_agents_dists[0] if len(near_agents_dists)>0 else (None, None)
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
        }'''
        
    def decide(self):
        if self.context['self_dna'].reproduction == 'mate':
            can_reproduce = self.context['nearest_ally'] is not None and self.context['ready_to_reproduce'] and self.context['nearest_ally'].ready_to_reproduce
        else: # clone
            can_reproduce = self.context['ready_to_reproduce'] # and self.context['self_dna'].energy_to_reproduce * self.energy_to_reproduce_threshold_ratio > self.context['energy']
        can_attack = self.context['nearest_enemy'] is not None
        can_flee = self.context['nearest_enemy'] is not None
        can_follow = self.context['nearest_ally'] is not None
        can_eat = self.context['nearest_food'] is not None 
        # last goal
        if self.memory_counter <=0:
            self.last_goal = None
        if self.last_goal is not None and not self.last_goal.completed:
            self.memory_counter -= 1
            if isinstance(self.last_goal, ReproduceGoal) and can_reproduce:
                return ReproduceGoal(self.context)
            if isinstance(self.last_goal, AttackGoal) and can_attack and not can_reproduce:
                return AttackGoal(self.context)
            if isinstance(self.last_goal, EatGoal) and can_eat and not can_reproduce:
                return EatGoal(self.context) 
        if self.context['self_dna'].reproduction == 'mate':  
            will_reproduce = can_reproduce and self.context['nearest_agent_dna_dist'] < self.context['self_dna'].partner_dna_distance
        else: # clone
            will_reproduce = can_reproduce
        will_attack = can_attack and np.random.rand() < self.context['self_dna'].aggressive and self.context['nearest_enemy_dna_dist'] > self.context['self_dna'].partner_dna_distance and self.context['nearest_enemy_dist'] < self.attack_distance
        will_flee = can_flee and self.context['nearest_enemy_dna_dist'] > self.context['self_dna'].partner_dna_distance and self.context['nearest_enemy_dist'] < self.flee_distance
        will_follow = can_follow and np.random.rand() < self.context['self_dna'].social and self.follow_min_distance < self.context['nearest_ally_dist'] < self.follow_max_distance
        will_eat = can_eat and self.context["energy"] < self.starve_threshold_ratio
        # mating
        goal = WanderGoal() 
        if will_reproduce:
            goal = ReproduceGoal(self.context)
        elif will_attack:
            goal = AttackGoal(self.context)
        elif will_flee:
            goal = FleeGoal(self.context)     
        elif will_eat:
            goal = EatGoal(self.context) 
        elif will_follow:
            goal = FollowGoal(self.context)
        self.last_goal = goal
        self.memory_counter = self.memory_time
        return goal