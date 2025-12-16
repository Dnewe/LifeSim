from typing import Dict, List
from multiprocessing import Queue
from world.world import World
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
        self.fps_list = []
        # general
        self.population = []
        self.births = []
        self.deaths = {}  
        # species
        self.species_scatter_data = np.zeros((2,2))
        self.species_dendrogram_data = None
        self.species_labels = []
        self.n_species = 1
        # genes 
        self.genes_mean_list = {f"mean_{g}": [0] for g in genes}
        self.genes_cv_list = {f"cv_{g}": [0] for g in genes}
        self.species_genes_mean = {s: [0] for s in range(self.n_species)}
        self.species_genes_cv = {s: [0] for s in range(self.n_species)}


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
        if len(world.agents) > 0:
            self.update_species_genes_data(world)
            self.update_species_scatter_data(world)
            self.update_species_dendrogram_data(world)
            # genes
            self.update_genes_metrics(world)
        
        self.queue.put(self.snapshot())
        
    def snapshot(self):
        return {
            # sim info 
            "n_step": self.n_step,
            "fps_cur": self.fps_list[-1],
            "fps_avg": np.mean(self.fps_list),
            # general
            "population": self.population[:],
            "births": self.births[:],
            "deaths": {**self.deaths},
            # species
            "n_species": self.n_species,
            "species_scatter_data_and_labels": (self.species_scatter_data, self.species_labels),
            "species_dendrogram_data_and_cutoff": (self.species_dendrogram_data, self.speciation_cutoff),
            # genes
            **{**self.genes_mean_list},
            **{**self.genes_cv_list}, # unpack copy of dict
            "species_genes_mean": self.species_genes_mean,
            "species_genes_cv": self.species_genes_cv,
        }
    
    def update_frame_per_s(self, length_running_avg=1000):
        cur_time = time.time_ns()
        v = self.step_freq * 1e9/(cur_time - self.prev_time + 1e-8)
        self.fps_list.append(v)
        if len(self.fps_list) > length_running_avg//self.step_freq:
            self.fps_list.pop(0)
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
        
    def update_species_genes_data(self, world: World):
        self.species_genes_mean = {s: [] for s in range(self.n_species)}
        self.species_genes_cv = {s: [] for s in range(self.n_species)}
        for s in range(self.n_species):
            means_dict = world.genetic_context.species_genes_mean[s]
            stds_dict = world.genetic_context.species_genes_std[s]
            means = [means_dict.get(g,0.) for g in self.genes]
            cvs = [stds_dict.get(g,0.) / means_dict.get(g,1e-8) for g in self.genes]
            self.species_genes_mean[s] = means
            self.species_genes_cv[s] = cvs
    
    def update_genes_metrics(self, world: World):
        means = world.genetic_context.genes_mean
        stds = world.genetic_context.genes_std
        for g in self.genes:
            self.genes_mean_list[f'mean_{g}'].append(means.get(g, 0))
            self.genes_cv_list[f'cv_{g}'].append(stds.get(g, 0)/means.get(g, 1e-8))
