from typing import TYPE_CHECKING, List
if TYPE_CHECKING:
    from agent.dna import DNA
    from agent.agent import Agent
import numpy as np



class GeneticContext():
    def __init__(self) -> None:
        self.genes_mean = {}
        self.genes_std = {}
            
    def dna_distance(self, dna1, dna2):
        acc = 0.0
        for gene in dna1.gene_values:
            v1 = dna1.gene_values[gene]
            v2 = dna2.gene_values[gene]
            # normalized difference
            delta = (v1 - v2) / self.genes_std[gene][1]
            acc += delta * delta
        return acc ** 0.5
        
    def update_stats(self, all_dna: List['DNA']):
        sums = {}
        sq_sums = {}
        count = 0     
        for dna in all_dna:
            count+= 1
            for k, v in dna.genes_values.items():
                sums[k] = sums.get(k, 0.0) + v
                sq_sums[k] = sq_sums.get(k, 0.0) + v*v               
        for k in sums:
            mean = sums[k] / count
            var = sq_sums[k] / count - mean*mean
            std = (var ** 0.5) + 1e-8
            self.genes_mean[k] = mean
            self.genes_std[k] = std
            
    def compute_species(self, agents):
        pass # TODO
        