import sys
from typing import Dict, Literal, Self
from utils.eval_utils import eval_expr
import numpy as np
import math


class DNA():
    # way to merge two parent dnas into a child dna
    merge_mode: Literal['mean', 'rdm_choice']
    # rules from config
    genes_rules: Dict
    attributes_rules: Dict
    # values
    genes_values: Dict
    # scale of the normal distrubition used during random mutation
    mutation_scale: float
    
    ### Attributes
    # physical
    size: float
    speed: float
    # 
    damage: float
    health: float
    # physiological
    lifespan: float
    max_energy: float
    energy_to_mate: float
    age_to_mate: float
    # sensory
    vision_range: float
    # energy costs
    step_cost: float
    sense_cost: float
    idle_cost: float
    # constants
    mating_cooldown: float
    
    
    def __init__(self, genes_values, genes_rules, attributes_rules, merge_mode, mutation_scale) -> None:
        self.genes_values = genes_values
        self.genes_rules = genes_rules
        self.attributes_rules = attributes_rules
        self.merge_mode = merge_mode
        self.mutation_scale = mutation_scale
        self.compute_attributes()
    
    @classmethod
    def from_config(cls, config: Dict):
        genes_values = {k: v['value'] for k, v in config['genes'].items()}
        return cls(genes_values, config['genes'], config['attributes'], config['merge_mode'], config['mutation_scale'])
    
    @classmethod
    def from_parents(cls, dna1:Self, dna2:Self):
        merge_mode = np.random.choice([dna1.merge_mode, dna2.merge_mode])
        mutation_scale = np.random.choice([dna1.mutation_scale, dna2.mutation_scale])
        genes_values = {k: cls._merge_values(dna1.genes_values[k], dna2.genes_values[k], merge_mode) for k in dna1.genes_values.keys()}
        return cls(genes_values, dna1.genes_rules, dna1.attributes_rules, merge_mode, mutation_scale)
    
    @classmethod
    def _merge_values(cls, v1, v2, merge_mode):
        if merge_mode == 'mean' and all(isinstance(v, (int, float)) for v in [v1, v2]):
            return np.mean([v1, v2])
        elif merge_mode == 'rdm_choice':
            return np.random.choice([v1, v2])
        else:
            print(f'merge mode "{merge_mode}" not implemented.')
            sys.exit(1)
            
    def compute_attributes(self):
        for name, rules in self.attributes_rules.items():
            value = rules["factor"] * eval_expr(rules["expr"], self.genes_values)
            value = self._clamp_value(value, rules["min"], rules["max"])
            setattr(self, name, value)
    
    def mutate(self, scale_factor=1.):
        # TODO other types of mutation
        for k, rules in self.genes_rules.items():
            if rules['mutable']:
                scale = rules['mutation_factor'] * self.mutation_scale * scale_factor
                if rules['type'] == 'continuous':
                    self.genes_values[k] = self._clamp_value(self._random_continuous(self.genes_values[k], scale), rules['min'], rules['max'])
                elif rules['type'] == 'discrete':
                    self.genes_values[k] = self._random_discrete(self.genes_values[k], scale, rules['domain'], rules['weights'])
                
    def _random_discrete(self, cur_v, scale, domain, weights):
        probs = [(scale*w)/(len(domain)-1) if cur_v == v else 1-scale for v,w in zip(domain, weights)]
        return np.random.choice(domain, p=probs)
        
    def _random_continuous(self, cur_v, scale):
        v = cur_v * (1 + np.random.normal(0, scale))
        return v
        
    def _clamp_value(self, v, minv, maxv):
        minv = minv if isinstance(minv, (int, float)) else -math.inf
        maxv =  maxv if isinstance(maxv, (int, float)) else math.inf
        return max(minv, min(v, maxv))
    
    
                