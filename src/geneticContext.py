from typing import TYPE_CHECKING, List, Dict
from collections import defaultdict
import numpy as np
if TYPE_CHECKING:
    from agent.agent import Agent
    from world.world import World
import utils.timeperf as timeperf
from agent.genome import Genome
from agent.brain.brain import Brain


class GeneticContext:
    def __init__(self, update_freq=50) -> None:
        self.update_freq = update_freq
        self.initialized = False
        # genes
        self.species_genes_mean = {} # k: sid, v: (k: gene, v: mean)
        self.species_genes_std = {} # k: sid, v: (k: gene, v: std)
        # brains
        self.brains_mean = {} # k: action, v: (k: weight, v: mean)
        self.brains_std = {} # k: action, v: (k: weight, v: std)
        self.species_brains_mean = defaultdict(dict) # k: sid, v: (k: action, v: (k: weight, v: mean))
        self.species_brains_std = defaultdict(dict) # k: sid, v: (k: action, v: (k: weight, v: std))
        
    @classmethod
    def from_config(cls, config):
        return cls(config['gc_update_freq'])
    
    @timeperf.timed()
    def update(self, world: 'World', global_only=False):
        if world.step_count % self.update_freq == 0 and world.n_agents > 0:
            self.update_gene_stats(world.agents, global_only)
            self.update_brain_stats(world.agents, global_only)
    
    def get_gene_mean(self, gene, species=None) -> float:
        if species is None:
            return self.genes_mean[Genome.key_to_idx[gene]]
        return self.species_genes_mean[species][Genome.key_to_idx[gene]]
            
    def get_gene_std(self, gene, species=None) -> float:
        if species is None:
            return self.genes_std[Genome.key_to_idx[gene]]
        return self.species_genes_std[species][Genome.key_to_idx[gene]]
                
    def get_brain_mean(self, key, species=None) -> float:
        if species is None:
            return self.brains_mean[Brain.key_to_idx[key]]
        return self.species_genes_mean[species][Brain.key_to_idx[key]]
            
    def get_brain_std(self, key, species=None) -> float:
        if species is None:
            return self.brains_std[Brain.key_to_idx[key]]
        return self.species_genes_std[species][Brain.key_to_idx[key]]
    
    def update_gene_stats(self, agents: List['Agent'], eps=1e-9, global_only=False):
        # Global
        X = np.stack([a.genome.to_vector() for a in agents])
        mean = X.mean(axis=0)
        std = X.std(axis=0) + eps
        self.genes_mean = mean
        self.genes_std = std
        
        if global_only:
            return
        # Per-species
        species_indices = defaultdict(list)
        for i, a in enumerate(agents):
            species_indices[a.species].append(i)
        for sid, idx in species_indices.items():
            Xs = X[idx]
            self.species_genes_mean[sid] = Xs.mean(axis=0)
            self.species_genes_std[sid]  = Xs.std(axis=0) + eps
    
    def update_brain_stats(self, agents: List['Agent'], eps=1e-9, global_only=False):
        B = np.stack([a.brain.to_vector() for a in agents])
        mean = B.mean(axis=0)
        std  = B.std(axis=0) + eps
        self.brains_mean = mean
        self.brains_std  = std
        
        if global_only:
            return
        # Per-species
        species_indices = defaultdict(list)
        for i, agt in enumerate(agents):
            species_indices[agt.species].append(i)

        for sid, idx in species_indices.items():
            Bs = B[idx]
            self.species_brains_mean[sid] = Bs.mean(axis=0)
            self.species_brains_std[sid]  = Bs.std(axis=0) + eps