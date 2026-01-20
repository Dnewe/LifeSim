from typing import TYPE_CHECKING, List, Dict
if TYPE_CHECKING:
    from agent.agent import Agent
    from world.world import World
    from agent.genome import Genome
    from agent.brain.brain import Brain
from collections import defaultdict
import numpy as np


class GeneticContext:
    def __init__(self, genes, actions_inputs) -> None:
        # gene index
        self.genes = list(genes)
        self.n_genes = len(self.genes)
        self.g_to_idx = {g: i for i,g in enumerate(self.genes)}
        # brain index
        self.actions_inputs = actions_inputs
        self.actions = list(actions_inputs.keys())
        self.n_actions = len(self.actions)
        self.actions_inp_to_idx = {a: {inp: i for i, inp in enumerate(actions_inputs[a])} for a in self.actions}
        # genes
        self.genes_mean = np.zeros(self.n_genes) # k: gene, v: mean
        self.genes_std = np.zeros(self.n_genes) # k: gene, v: std
        self.species_genes_mean = {} # k: sid, v: (k: gene, v: mean)
        self.species_genes_std = {} # k: sid, v: (k: gene, v: std)
        # brains
        self.brains_mean = {} # k: action, v: (k: weight, v: mean)
        self.brains_std = {} # k: action, v: (k: weight, v: std)
        self.species_brains_mean = defaultdict(dict) # k: sid, v: (k: action, v: (k: weight, v: mean))
        self.species_brains_std = defaultdict(dict) # k: sid, v: (k: action, v: (k: weight, v: std))
        
    @classmethod
    def from_config(cls, config):
        genes = config['genome']['genes'].keys()
        actions_inputs = {a: [inp for inp in inputs] for a, inputs in config['brain']['utility_scores'].items()}
        return cls(genes, actions_inputs)
    
    def update(self, world: 'World', global_only=False):
        if world.n_agents > 0:
            self.update_gene_stats(world.agents, global_only)
            self.update_brain_stats(world.agents, global_only)
    
    def get_gene_mean(self, gene, species=None):
        if species is None:
            return self.genes_mean[self.g_to_idx[gene]]
        return self.species_genes_mean[species][self.g_to_idx[gene]]
            
    def get_gene_std(self, gene, species=None):
        if species is None:
            return self.genes_std[self.g_to_idx[gene]]
        return self.species_genes_std[species][self.g_to_idx[gene]]
                
    def get_action_input_mean(self, action, inp, species=None):
        if species is None:
            return self.brains_mean[action][self.actions_inp_to_idx[action][inp]]
        return self.species_genes_mean[species][action][self.actions_inp_to_idx[action][inp]]
            
    def get_action_input_std(self, action, inp, species=None):
        if species is None:
            return self.brains_std[action][self.actions_inp_to_idx[action][inp]]
        return self.species_genes_std[species][action][self.actions_inp_to_idx[action][inp]]
    
    def update_gene_stats(self, agents: List['Agent'], gamma_mean=0.4, gamma_std=0.8, eps=1e-9, global_only=False):
        # Global
        X = np.stack([self._genome_to_vector(a.genome) for a in agents])
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
    
    def update_brain_stats(self, agents: List['Agent'], gamma_mean=0.4, gamma_std=0.8, eps=1e-9, global_only=False):
        # Stack brains
        for a in self.actions:
            B = np.stack([self._weights_to_vector(a, agt.brain.get_data()[a]) for agt in agents])  # (N, W)
            mean = B.mean(axis=0)
            std  = B.std(axis=0) + eps
            self.brains_mean[a] = mean
            self.brains_std[a]  = std

            # Per-species
            species_indices = defaultdict(list)
            for i, agt in enumerate(agents):
                species_indices[agt.species].append(i)

            for sid, idx in species_indices.items():
                Bs = B[idx]
                self.species_brains_mean[sid][a] = Bs.mean(axis=0)
                self.species_brains_std[sid][a]  = Bs.std(axis=0) + eps
            
    def _genome_to_vector(self, genome: 'Genome'):
        v = np.empty(self.n_genes, dtype=np.float32)
        for g, i in self.g_to_idx.items():
            v[i] = genome.gene_values[g]
        return v
    
    def _weights_to_vector(self, action, weights):
        v = np.empty(len(weights), dtype=np.float32)
        for inp, i in self.actions_inp_to_idx[action].items():
            v[i] = weights[inp]
        return v