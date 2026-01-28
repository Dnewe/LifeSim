import numpy as np
from typing import Dict, List, TYPE_CHECKING, Self
if TYPE_CHECKING:
    from world.world import World
    from sprite.agent import Agent
from sprite.brain.brain import Brain
from sprite.genome import Genome
import sprite.goal as goal


MIN_G = 0.01
MAX_G = 0.9    
MIN_W = -5
MAX_W = +5 
GAMMA_SCALE_FACTOR = 0.5


class UtilityScore:
    action: str
    
    def __init__(self, action: str, inputs_weights:Dict[str,float]) -> None:
        self.action = action
        self.inputs_weights = dict(inputs_weights)
        self.prev_u = 0
    
    def compute(self, inputs:Dict):
        u = self.inputs_weights.get('bias', 0)
        for k,v in inputs.items():
            w = self.inputs_weights.get(k, 0)
            u += w * v
        g = self.inputs_weights.get('gamma', 0)
        u = g * self.prev_u + (1-g) * u 
        self.prev_u = u
        return u
    
    def randomize(self, n_weights:int =-1, scale=0.1):
        # TODO update according to heuristics 
        weights_to_change = np.random.choice(list(self.inputs_weights), n_weights) if n_weights >0 else self.inputs_weights.keys()
        for k in weights_to_change:
            if k == 'gamma':
                delta = np.random.normal(0, GAMMA_SCALE_FACTOR * scale / 6)
                g = self.inputs_weights[k] + delta
                self.inputs_weights[k] = max(min(g, MAX_G), MIN_G)
            else:
                delta = np.random.normal(0, (MAX_W - MIN_W) * scale / 6)
                self.inputs_weights[k] += delta
                self.inputs_weights[k] = max(min(self.inputs_weights[k], MAX_W), MIN_W)
            

class UtilityFunBrain(Brain):
    initialized = False
    
    @classmethod
    def initialize(cls, config):
        cls.actions = sorted(config['utility_scores'].keys())
        cls.n_actions = len(cls.actions)
        Brain.key_to_idx = {}
        counter = 0
        for a in cls.actions:
            for k in config['utility_scores'][a].keys():
                Brain.key_to_idx[(a,k)] = counter
                counter += 1
        Brain.n_keys = len(Brain.key_to_idx)
        cls.initialized = True

    def __init__(self, utility_scores: List[UtilityScore], mutation_scale, n_mutations) -> None:
        self.utility_scores = utility_scores
        self.mutation_scale = mutation_scale
        self.n_mutations = n_mutations
        
    @classmethod
    def from_config(cls, config):
        if not cls.initialized:
            cls.initialize(config)
        utility_scores = []
        for k, v in config['utility_scores'].items():
            utility_scores.append(UtilityScore(k, v))
        return cls(utility_scores, config['mutation_scale'], config['n_mutations'])    
    
    def sense(self, agent, world):
        super().sense(agent, world)    
        self.inputs = self.get_inputs(agent, world)
    
    def decide(self):
        best_action = None
        best_score = float("-inf")
        best_args = None
        for f in self.utility_scores:
            action = f.action
            score = f.compute(self.inputs)
            if score <= best_score:
                continue
            args = self.get_action_args(action)
            if goal.valid_action_args(action, args):
                best_score = score
                best_action = action
                best_args = args
        self.context = {} # reset context to avoid being copied when cloned
        if best_action is None:
            return "idle", None
        return best_action, best_args
            
    def mutate(self, scale_factor = 1., n_mutations=None):
        scale = self.mutation_scale * scale_factor
        n_weights = self.n_mutations if n_mutations is None else n_mutations
        for us in self.utility_scores:
            us.randomize(n_weights, scale)
            
    def get_data(self):
        data = {}
        for us in self.utility_scores:
            data[us.action] = us.inputs_weights
        return data
    
    def get_action_args(self, action):
        args_keys = goal.GOALS_MAP[action].args_keys
        args = {}
        for k in args_keys:
            arg = None
            match k:
                case 'food_pos': arg = self.context['nearest_food_pos']
                case 'food_dist': arg = self.context['nearest_food_dist']
                case 'food_energy': arg = self.context['nearest_food_energy']
                case 'food_size': arg = self.context['nearest_food_size']
                case 'target': arg = self.context['near_agents'][0][0] if len(self.context['near_agents'])>0 else None  # nearest agent
                case 'target_dist': arg = self.context['near_agents'][0][1] if len(self.context['near_agents'])>0 else 1  # nearest agent dist
                case 'can_reproduce': arg = self.context['self_agent'].can_reproduce
            args[k] = arg
        return args
    
    def get_inputs(self, agent: 'Agent', world: 'World'):
        agents = self.context['near_agents']
        food_dist = self.context['nearest_food_dist']
        food_energy = self.context['nearest_food_energy']
        return {
            # agent variables
            'energy': agent.energy / agent.genome.max_energy - 0.5,
            'health': agent.health / agent.genome.max_health - 0.5,
            'age': agent.age / agent.genome.max_age - 0.5,
            # environment
            'n_agents': len(agents) / 5 - 0.5,
            'agent_proximity': 1 - agents[0][1] / agent.genome.vision_range - 0.5 if len(agents)>0 else 0.5,
            'agent_genome_dist': Genome.distance(agent.genome, agents[0][0].genome, world.gc.genes_std) if len(agents)>0 else 0,
            'food_proximity': 1 - food_dist / agent.genome.vision_range - 0.5,
            'food_energy': food_energy / world.foodmap.food_base_energy - 0.5,
        }
        
    def to_vector(self) -> np.ndarray:
        vec = np.empty(self.n_keys, dtype=np.float32)
        for u in self.utility_scores:
            for k,v in u.inputs_weights.items():
                vec[self.key_to_idx[(u.action, k)]] = v
        return vec
    
    