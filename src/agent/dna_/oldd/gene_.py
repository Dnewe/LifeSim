from typing import Dict, Self, Literal
from agent.dna.oldd.attribute import *
import numpy as np
import sys


OPS = {
    "min":    lambda v, arg: max(arg, v),
    "max":    lambda v, arg: min(arg, v),
    "factor": lambda v, arg: v * arg,
    "offset": lambda v, arg: v + arg,
    "log":    lambda v, arg: np.log(v, arg),
    "pow":    lambda v, arg: np.pow(v, arg),
    "sup":    lambda v, arg: v > arg,
    "inf":    lambda v, arg: v < arg,
    "if":     lambda v, arg: arg[0] if v else arg[1],
    "floor":  lambda v, _  : np.floor(v)
}


class Gene():
    # value of the gene
    value: float|str|bool
    # attr dict (attr_name, attr_value)
    attributes_rules: Dict
    attributes: Dict[str, float|str|bool]
    # whether it is mutable or not
    mutable: bool
    
    def __init__(self, value, attributes_rules, mutable:bool, mutation_scale_factor:float) -> None:
        self.value = value
        self.mutable = mutable
        self.mutation_scale_factor = mutation_scale_factor
        self.attributes_rules = attributes_rules
        self.attributes = {}
        
    @classmethod
    def from_config(cls, config):
        if config['type'] == 'discrete':
            return DiscreteGene.from_config(config)
        elif config['type'] == 'continuous':
            return ContinuousGene.from_config(config)
        else:
            print(f'Gene of type "{config['type']}" not implemented')
            sys.exit(1)
    
    @classmethod
    def from_parents(cls, gene1: Self, gene2: Self, merge_mode: str):
        pass
    
    def compute_attributes(self):
        for k, rules in self.attributes_rules.items():
            v = self.value
            for r in rules:
                op, arg = r
                v = OPS[op](v, arg)
            self.attributes[k] = v
                
    def random_mutation(self, scale):
        pass



class ContinuousGene(Gene):
    def __init__(self, value, attributes, mutable: bool, mutation_scale_factor:float, minv:float|None, maxv:float|None) -> None:
        super().__init__(value, attributes, mutable, mutation_scale_factor)
        self.minv = minv if isinstance(minv, (float, int)) else None
        self.maxv = maxv if isinstance(maxv, (float, int)) else None
        self._clamp_value()
        
    @classmethod
    def from_config(cls, config):
        return cls(config['default'], 
                   config['attributes'], 
                   config['mutable'], 
                   config['mutation_scale_factor'], 
                   config['min'], 
                   config['max'])
        
    def random_mutation(self, scale):
        scale = scale * self.mutation_scale_factor
        self.value = self.value * (1 + np.random.normal(0, scale))
        self._clamp_value()
            
    def _clamp_value(self):
        if self.minv is not None:
            self.value = max(self.minv, self.value)
        if self.maxv is not None:
            self.value = min(self.maxv, self.value)    



class DiscreteGene(Gene):
    def __init__(self, value, domain: List, weights: List[float], attributes, mutable: bool, mutation_scale_factor:float) -> None:
        super().__init__(value, attributes, mutable, mutation_scale_factor)
        self.domain = domain 
        total_w = np.sum(weights)
        self.weights = [w*len(weights) / total_w for w in weights]
        
    @classmethod
    def from_config(cls, config):
        return cls(config['default'], 
                   config['attributes'], 
                   config['weights'],
                   config['domain'],
                   config['mutable'], 
                   config['mutation_scale_factor']
                   )
        
    def random_mutation(self, scale):
        scale = scale * self.mutation_scale_factor
        probs = [(scale*w)/(len(self.domain)-1) if self.value == v else 1-scale for v,w in zip(self.domain, self.weights)]
        self.value = np.random.choice(self.domain, p=probs)