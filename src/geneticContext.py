from typing import TYPE_CHECKING, List
if TYPE_CHECKING:
    from agent.dna import DNA
    from agent.agent import Agent
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
import numpy as np



class GeneticContext():
    def __init__(self, speciation_cutoff = 4.0) -> None:
        self.genes_mean = {}
        self.genes_std = {}
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
        
    def update_stats(self, all_dna: List['DNA']):
        sums = {}
        sq_sums = {}
        count = 0     
        for dna in all_dna:
            count+= 1
            for k, v in dna.gene_values.items():
                sums[k] = sums.get(k, 0.0) + v
                sq_sums[k] = sq_sums.get(k, 0.0) + v*v               
        for k in sums:
            mean = sums[k] / count
            var = sq_sums[k] / count - mean*mean
            std = (var ** 0.5) + 1e-8
            self.genes_mean[k] = mean
            self.genes_std[k] = std
            
    def compute_species(self, agents: List['Agent']):
        dna_distances = np.zeros((len(agents), len(agents)))
        for i, agent1 in enumerate(agents):
            for j, agent2 in enumerate(agents):
                if agent1 == agent2:
                    continue
                dna_distances[i,j] = self.dna_distance(agent1.dna, agent2.dna)
        
        labels = self.model.fit_predict(dna_distances)
        for i, agent in enumerate(agents):
            agent.specie = labels[i]
        
        