import numpy as np
from typing import Dict, List
from agent.brain.brain import Brain
import agent.goal as goal


MIN_G = 0.01
MAX_G = 0.9    
MIN_W = -5
MAX_W = +5 


class UtilityScore:
    action: str
    
    def __init__(self, action: str, context_weights:Dict[str,float]) -> None:
        self.action = action
        self.context_weights = dict(context_weights)
        self.prev_u = 0
    
    def compute(self, context:Dict):
        u = 0
        for k,v in context.items():
            if isinstance(v, (int, float)):
                w = max(min(self.context_weights.get(k, 0), MAX_G), MIN_W)
                u += w * v
        u += self.context_weights.get('bias', 0)
        g = self.context_weights.get('gamma', 0)
        u = g * self.prev_u + (1-g) * u 
        self.prev_u = u
        return u
    
    def randomize(self, n_weights:int =-1, scale=0.1):
        # TODO update according to heuristics 
        weights_to_change = np.random.choice(list(self.context_weights), n_weights) if n_weights >0 else self.context_weights.keys()
        for k in weights_to_change:
            if k == 'gamma':
                delta = np.random.normal(0, 0.1 * scale)
                self.context_weights[k] += delta
                self.context_weights[k] = max(min(self.context_weights[k], MAX_G), MIN_G)
            else:
                delta = np.random.normal(0, (MAX_W - MIN_W) * scale / 6)
                self.context_weights[k] += delta
                self.context_weights[k] = max(min(self.context_weights[k], MAX_W), MIN_W)
            self.context_weights[k] += (np.random.normal(0, scale)) * (MAX_W - MIN_W) * scale
            self.context_weights[k] = max(min(self.context_weights[k], MAX_W), MIN_W)
            

class UtilityFunBrain(Brain):
    def __init__(self, utility_scores: List[UtilityScore], mutation_scale, n_mutations) -> None:
        self.utility_scores = utility_scores
        self.mutation_scale = mutation_scale
        self.n_mutations = n_mutations
        
    @classmethod
    def from_config(cls, config):
        utility_scores = []
        for k, v in config['utility_scores'].items():
            utility_scores.append(UtilityScore(k, v))
        return cls(utility_scores, config['mutation_scale'], config['n_mutations'])        
    
    def decide(self):
        actions_and_scores = [(f.action, f.compute(self.context)) for f in self.utility_scores]
        sorted_actions = [x[0] for x in sorted(actions_and_scores, key= lambda x: x[1], reverse=True)]
        while not goal.valid_action(sorted_actions[0], self.context):
            sorted_actions.pop(0)
        self.action = sorted_actions[0] if len(sorted_actions)>0 else 'idle'
        return self.action
            
    def mutate(self, scale_factor = 1., n_mutations=None):
        scale = self.mutation_scale * scale_factor
        n_weights = self.n_mutations if n_mutations is None else n_mutations
        for us in self.utility_scores:
            us.randomize(n_weights, scale)
            
    def get_data(self):
        data = {}
        for us in self.utility_scores:
            data[us.action] = us.context_weights
        return data
    