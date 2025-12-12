from typing import Dict, List
from multiprocessing import Queue
from world import World
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import linkage
import numpy as np
import time


class Metrics():
    def __init__(self, genes, step_freq=60) -> None:
        self.genes = genes
        self.queue = Queue()
        self.step_freq = step_freq
        self.prev_time = 0
        # sim info
        self.n_step = 0
        self.step_per_s = 0
        # general
        self.population = []
        self.births = []
        self.deaths = {}  
        # genes 
        self.genes_mean_values = {f"mean_{g}": [] for g in genes}
        self.genes_cv_values = {f"cv_{g}": [] for g in genes}
        # species
        self.species_scatter_data = np.zeros((2,2))
        self.species_dendrogram_data = None
        self.species_labels = []
        self.n_species = 1


    def update(self, world: World):
        self.n_step = world.step_count
        if self.n_step % self.step_freq != 0:
            return
        # general
        self.update_frame_per_s()
        self.population.append(world.n_agents)
        self.births.append(world.n_births)
        for type, n in world.n_deaths_per_type.items():
            if type not in self.deaths:
                self.deaths[type] = []
            else: 
                self.deaths[type].append(n)
        # species
        self.n_species = world.genetic_context.n_species
        self.update_species_scatter_data(world)
        self.update_species_dendrogram_data(world)
        # genes
        self.update_genes_metrics(world)
        
        self.queue.put(self.snapshot())
        
    def snapshot(self):
        return {
            # sim info 
            "n_step": self.n_step,
            "step_per_s": self.step_per_s,
            # general
            "population": self.population[:],
            "births": self.births[:],
            "deaths": {**self.deaths},
            # species
            "n_species": self.n_species,
            "species_scatter_data_and_labels": (self.species_scatter_data, self.species_labels),
            "species_dendrogram_data_and_cutoff": (self.species_dendrogram_data, self.speciation_cutoff),
            # genes
            **{**self.genes_mean_values},
            **{**self.genes_cv_values} # unpack copy of dict
        }
    
    def update_frame_per_s(self):
        cur_time = time.time_ns()
        self.step_per_s = self.step_freq * 1e9/(cur_time - self.prev_time)
        self.prev_time = cur_time

    def update_species_scatter_data(self, world: World):
        if len(world.agents) < 2:
            self.species_scatter_data = np.zeros((2,2))
            return
        dnas = [a.dna for a in world.agents]
        genes = dnas[0].gene_values.keys()
        mat = np.array([[dna.gene_values[g] for g in genes] for dna in dnas])
        pca = PCA(n_components=2)
        self.species_labels = np.array([a.specie for a in world.agents])
        self.species_scatter_data = pca.fit_transform(mat)
        
    def update_species_dendrogram_data(self, world: World):
        distance_matrix = world.genetic_context.distance_matrix
        condensed = distance_matrix[np.triu_indices(len(distance_matrix), k=1)]
        self.species_dendrogram_data = linkage(condensed, method='average')
        self.speciation_cutoff = world.genetic_context.speciation_cutoff
    
    def update_genes_metrics(self, world: World):
        means = world.genetic_context.genes_mean
        stds = world.genetic_context.genes_std
        for g in self.genes:
            self.genes_mean_values[f'mean_{g}'].append(means[g])
            self.genes_cv_values[f'cv_{g}'].append(stds[g]/means[g])
