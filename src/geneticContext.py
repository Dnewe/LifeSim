from typing import TYPE_CHECKING, List
if TYPE_CHECKING:
    from agent.dna import DNA
    from agent.agent import Agent
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
import numpy as np



class GeneticContext():
    def __init__(self, speciation_cutoff = 4.0) -> None:
        # genes
        self.genes_mean = {}
        self.genes_std = {}
        self.genes_mean_per_specie = {}
        self.genes_std_per_specie = {}
        # species
        self.speciation_cutoff = speciation_cutoff
        self.distance_matrix = np.zeros((2, 2))
        self.n_species = 1
        self.model = AgglomerativeClustering(
            n_clusters=None,
            metric='precomputed',
            distance_threshold=speciation_cutoff,
            linkage='average'
        )
            
    def dna_distance(self, dna1: 'DNA', dna2: 'DNA'):
        acc = 0.0
        for gene in dna1.gene_values.keys():
            v1 = dna1.gene_values[gene]
            v2 = dna2.gene_values[gene]
            # normalized difference
            delta = (v1 - v2) / self.genes_std[gene]
            acc += delta * delta
        return acc ** 0.5
        
    def update_stats(self, agents: List['Agent']):
        ### accumulators
        # global
        sums = {}
        sq_sums = {}
        count = 0  
        # per specie   
        sums_specie = {}
        sq_sums_specie = {} 
        count_specie = [0] * self.n_species
        ### accumulate
        for a in agents:
            s = a.specie
            count += 1
            count_specie[s] += 1
            for k, v in a.dna.gene_values.items():
                # global
                sums[k] = sums.get(k, 0.0) + v
                sq_sums[k] = sq_sums.get(k, 0.0) + v*v      
                # per specie 
                if k not in sums_specie:
                    sums_specie[k] = [0.0] * self.n_species
                    sq_sums_specie[k] = [0.0] * self.n_species
                sums_specie[k][s] += v
                sq_sums_specie[k][s] += v*v    
        ### compute stats            
        for k in sums:
            # global
            mean = sums[k] / count
            var = sq_sums[k] / count - mean*mean
            std = (var ** 0.5) + 1e-8
            self.genes_mean[k] = mean
            self.genes_std[k] = std
            # per specie
            if k not in self.genes_mean_per_specie:
                self.genes_mean_per_specie[k] = [0.0] * self.n_species
                self.genes_std_per_specie[k] = [0.0] * self.n_species
                for s in range(self.n_species):
                    if count_specie[s] == 0: # if empty specie
                        self.genes_mean_per_specie[k][s] = 0.0
                        self.genes_std_per_specie[k][s]  = 0.0
                        continue
                    m = sums_specie[k][s] / count_specie[s]
                    v = (sq_sums_specie[k][s] / count_specie[s]) - m*m
                    sd = (v ** 0.5) + 1e-8
                    self.genes_mean_per_specie[k][s] = m
                    self.genes_std_per_specie[k][s]  = sd
            
    def compute_species(self, agents: List['Agent']):
        if len(agents) <2:
            return
        distance_matrix = np.zeros((len(agents), len(agents)))
        for i, agent1 in enumerate(agents):
            for j, agent2 in enumerate(agents):
                if agent1 == agent2:
                    continue
                distance_matrix[i,j] = self.dna_distance(agent1.dna, agent2.dna)
        
        labels = self.model.fit_predict(distance_matrix)
        self.distance_matrix = distance_matrix
        self.n_species = len(np.unique(labels))
        for i, agent in enumerate(agents):
            agent.specie = labels[i]
        
        