import sys
from typing import Dict, Literal, Self, List
from utils.eval_utils import eval_expr
import numpy as np
import math
from copy import deepcopy
import utils.timeperf as timeperf


class Genome():
    initialized = False
    # way to merge two parent genomes into a child genome
    merge_mode: Literal['mean', 'rdm_choice']
    reproduction: Literal['mate', 'clone']
    # rules from config
    gene_rules: Dict
    attributes_rules: Dict
    # values
    gene_values: Dict
    # scale of the normal distrubition used during random mutation
    mutation_scale: float
    
    ### Attributes
    # physical
    size: float
    speed: float
    # behavioral
    aggressive: float
    social: float
    partner_genome_distance: float
    # physiological (attack & defense)
    damage: float
    max_health: float
    regeneration: float
    # physiological (energy & mating)
    max_age: float
    max_energy: float
    energy_to_reproduce: float
    maturity_age: float
    # alimentation
    metabolism_efficiency: float
    # sensory
    vision_range: float
    # energy costs
    step_cost: float
    sense_cost: float
    idle_cost: float
    # constants
    reproduce_cooldown: float
    
    @classmethod
    def initialize(cls, config):
        cls.genes = sorted(config['genes'].keys())
        cls.n_genes = len(cls.genes)
        cls.key_to_idx = {g: i for i,g in enumerate(cls.genes)}
        cls.initialized = True
    
    def __init__(self, gene_values, gene_rules, attributes_rules, reproduction, merge_mode, mutation_scale, n_mutations) -> None:
        self.gene_values = gene_values
        self.gene_rules = gene_rules
        self.attributes_rules = attributes_rules
        self.reproduction = reproduction
        self.merge_mode = merge_mode
        self.mutation_scale = mutation_scale
        self.n_mutations = n_mutations
        self.compute_attributes()
    
    @classmethod
    def from_config(cls, config: Dict):
        if not cls.initialized: 
            cls.initialize(config)
        gene_values = {k: v['value'] for k, v in config['genes'].items()}
        return cls(gene_values, config['genes'], config['attributes'], config['reproduction'], config['merge_mode'], config['mutation_scale'], config['n_mutations'])
    
    @classmethod
    def clone(cls, other):
        return deepcopy(other)
    
    @classmethod
    def from_parents(cls, genome1:Self, genome2:Self):
        merge_mode = np.random.choice([genome1.merge_mode, genome2.merge_mode])
        mutation_scale = np.random.choice([genome1.mutation_scale, genome2.mutation_scale])
        n_mutations = np.random.choice([genome1.n_mutations, genome2.n_mutations])
        gene_values = {k: cls._merge_values(genome1.gene_values[k], genome2.gene_values[k], merge_mode) for k in genome1.gene_values.keys()}
        return cls(gene_values, genome1.gene_rules, genome1.attributes_rules, genome1.reproduction, merge_mode, mutation_scale, n_mutations)
    
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
            value = rules["factor"] * eval_expr(rules["expr"], self.gene_values)
            value = self._clamp_value(value, rules["min"], rules["max"])
            setattr(self, name, value)
    
    def mutate(self, scale_factor= 1., n_mutations: int|None= None):
        # TODO other types of mutation
        if n_mutations is None:
            genes = np.random.choice(list(self.gene_rules.keys()), self.n_mutations) 
        elif n_mutations < 0:
            genes = self.gene_rules.keys()
        else:
            genes = np.random.choice(list(self.gene_rules.keys()), n_mutations)
        for k in genes:
            rules = self.gene_rules[k]
            if rules['mutable']:
                scale = rules['mutation_factor'] * self.mutation_scale * scale_factor
                if rules['type'] == 'continuous':
                    self.gene_values[k] = self._clamp_value(self._random_continuous(self.gene_values[k], scale), rules['min'], rules['max'])
                elif rules['type'] == 'discrete':
                    self.gene_values[k] = self._random_discrete(self.gene_values[k], scale, rules['domain'], rules['weights'])
    
    @classmethod
    @timeperf.timed()                
    def distance(cls, genome1: Self, genome2: Self, genes_std, alpha:float =1., eps:float =1e-9) -> float:
        v1 = np.log1p(genome1.to_vector())
        v2 = np.log1p(genome2.to_vector())
        weights = np.array([1.0 for k in cls.genes]) # TODO weights for genes
        # normalized euclidean
        diff = (v1 - v2) / (genes_std + eps)
        euclid = np.sqrt(np.sum(weights * diff * diff))
        # cosine distance (directional)
        cosine = 1.0 - np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + eps)
        d = euclid * (1.0 + alpha * cosine)  
        d /= cls.n_genes
        return d
                    
    def to_vector(self):
        vect = np.empty(self.n_genes, dtype=np.float32)
        for g, i in self.key_to_idx.items():
            vect[i] = self.gene_values[g]
        return vect
    
    def get_key_to_idx(self):
        keys = sorted(self.gene_values.keys())
        key_to_idx = {k: i for i,k in enumerate(keys)}
        return key_to_idx
                
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
    
    
                