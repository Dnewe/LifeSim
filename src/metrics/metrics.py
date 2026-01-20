from typing import Dict, List
from multiprocessing import Queue
from world.world import World
from sklearn.decomposition import PCA, IncrementalPCA
from scipy.cluster.hierarchy import linkage
from collections import defaultdict

import numpy as np
import time


class Metrics():
    def __init__(self, step_freq=60) -> None:
        self.queue = Queue(maxsize=1)
        self.step_freq = step_freq
        self.initialized = False

    def _init_vars(self, world: World):
        # sim info
        self.prev_time = 0
        self.n_step = 0
        self.fps_list = []
        # general
        self.population = []
        self.births = []
        self.deaths = {"age": [], "starvation": [], "killed": []}  
        # species
        self.species_scatter_data = None
        self.species_density_data = {'count': [], 'species': [], 'step': []}
        self.species_labels = []
        # genes 
        self.genes = world.gc.genes # helper
        self.genes_means = defaultdict(list)
        self.genes_cvs = defaultdict(list)
        self.species_genes_mean = {}
        self.species_genes_cv = {}
        # brain
        self.actions_inputs = world.gc.actions_inputs
        self.brains_means = defaultdict(lambda: defaultdict(list)) #{a: {w: []}}
        self.brains_cvs = defaultdict(lambda: defaultdict(list)) # {a: {w: []}}
        self.species_brains_mean = {}
        self.species_brains_cv = {}
        # action
        self.actions_density_data = {'count': [], 'action': [], 'step': []}

    def update(self, world: World):
        if not self.initialized:
            self._init_vars(world)
            self.initialized = True
    
        self.n_step = world.step_count
        if self.n_step % self.step_freq != 0:
            return
        # general
        self.update_frame_per_s()
        self.population.append(world.n_agents)
        self.births.append(world.n_births)
        for type, n in world.n_deaths_per_type.items():
            if type not in self.deaths:
                self.deaths[type] = [n]
            else: 
                self.deaths[type].append(n)
        # species
        self.n_species = world.speciator.n_species
        self.update_species_genes_data(world)
        self.update_species_scatter_data(world)
        self.update_species_density_data(world)
        # genes
        self.update_genes_metrics(world)
        # brain
        self.update_actions_density_data(world)
        self.update_brain_metrics(world)

        if self.queue.full():
            self.queue.empty()
        self.queue.put(self.snapshot())
        
    def snapshot(self):
        return {
            # sim info 
            "n_step": self.n_step,
            "fps_cur": self.fps_list[-1],
            "fps_avg": np.mean(self.fps_list),
            # general
            "steps": np.arange(0, self.n_step, step=self.step_freq),
            "population": self.population[:],
            "births": self.births[:],
            "deaths": {**self.deaths},
            # species
            "n_species": self.n_species,
            "species_scatter_data": self.species_scatter_data,
            "species_density_data": self.species_density_data,
            # genes
            "genes_means": {**self.genes_means},
            "genes_cvs": {**self.genes_cvs},
            "species_genes_mean": self.species_genes_mean,
            "species_genes_cv": self.species_genes_cv,
            # brains
            "actions_density_data": self.actions_density_data,
            "brains_means": {**self.brains_means},
            "brains_cvs": {**self.brains_cvs},
            "species_brains_mean": self.species_brains_mean,
            "species_brains_cv": self.species_brains_cv,
        }
    
    def update_frame_per_s(self, length_running_avg=1000):
        cur_time = time.time_ns()
        v = self.step_freq * 1e9/(cur_time - self.prev_time + 1e-8)
        self.fps_list.append(v)
        if len(self.fps_list) > length_running_avg//self.step_freq:
            self.fps_list.pop(0)
        self.prev_time = cur_time

    def update_species_scatter_data(self, world: World):
        if len(world.agents) <= 2:
            self.species_scatter_data = None
            return
        genomes_all = [a.genome for a in world.agents]
        genomes_repr = [s.genome for s in world.speciator.species]
        genes = genomes_all[0].gene_values.keys()
        mat_all = np.array([[genome.gene_values[g] for g in genes] for genome in genomes_all if genome not in genomes_repr])
        mat_repr = np.array([[genome.gene_values[g] for g in genes] for genome in genomes_repr])
        if mat_all.shape[0] < 2:
            return  # exit if not enough non-representant agents
        pca = IncrementalPCA(n_components=2, batch_size=256)
        points_all = pca.fit_transform(mat_all)
        points_repr = pca.transform(mat_repr)
        labels_all = np.array([int(a.species) for a in world.agents if a.genome not in genomes_repr])
        labels_repr = np.array([s.id for s in world.speciator.species])
        self.species_scatter_data = {'points_list': [points_all, points_repr],
                                     'labels_list': [labels_all, labels_repr],
                                     'markers': ['.', '*']}
        
    def update_species_density_data(self, world: World):
        step = world.step_count
        for sid, count in world.n_agents_per_species.items():
            self.species_density_data['count'].append(count)
            self.species_density_data['species'].append(sid)
            self.species_density_data['step'].append(step)
    
    def update_genes_metrics(self, world: World):
        for g in self.genes:
            mean = world.gc.get_gene_mean(g)
            std = world.gc.get_gene_std(g)
            self.genes_means[g].append(mean)
            self.genes_cvs[g].append(std / mean +1e-8)
        
    def update_species_genes_data(self, world: World):
        self.species_genes_mean = {s.id: [] for s in world.speciator.species}
        self.species_genes_cv = {s.id: [] for s in world.speciator.species}
        for s in world.speciator.species:
            sid = s.id
            means = [world.gc.get_gene_mean(g, sid) for g in self.genes]
            stds = [world.gc.get_gene_std(g, sid) for g in self.genes]
            cvs = [std / mean for std, mean in zip(stds, means)]
            #means_dict = world.gc.species_genes_mean[sid]
            #stds_dict = world.gc.species_genes_std[sid]
            #means = [means_dict[g] for g in self.genes]
            #cvs = [stds_dict[g] / means_dict[g] for g in self.genes]
            self.species_genes_mean[sid] = means
            self.species_genes_cv[sid] = cvs
            
    def update_actions_density_data(self, world: World):
        step = world.step_count
        for action, count in world.n_actions_per_type.items():
            self.actions_density_data['count'].append(count)
            self.actions_density_data['action'].append(action)
            self.actions_density_data['step'].append(step)
            world.n_actions_per_type[action] = 0 # reset count
            
    def update_brain_metrics(self, world: World):
        for a, inputs in self.actions_inputs.items():
            for inp in inputs:
                mean = world.gc.get_action_input_mean(a, inp)
                std = world.gc.get_action_input_std(a, inp)
                self.brains_means[a][inp].append(mean)
                self.brains_cvs[a][inp].append(std / mean + 1e-8)
            
    def update_species_brains_data(self, world: World):
        self.species_brains_mean = {s.id: {} for s in world.speciator.species}
        self.species_brains_cv = {s.id: {} for s in world.speciator.species}
        for s in world.speciator.species:
            sid = s.id
            #means_dict = world.gc.species_brains_mean[sid]
            #stds_dict = world.gc.species_brains_std[sid]
            means = defaultdict(dict)
            cvs = defaultdict(dict)
            for a, inputs in self.actions_inputs.items():
                for inp in inputs:
                    mean = world.gc.get_action_input_mean(a, inp)
                    std = world.gc.get_action_input_std(a, inp)
                    means[a][inp] = mean
                    cvs[a][inp] = std / mean + 1e-8
                #means[a] = {w: means_dict[a][w] for a in means_dict for w in means_dict[a]}
                #cvs[a] = {w: stds_dict[a][w]  / (means_dict[a][w] + 1e-8)  for a in means_dict for w in means_dict[a]}
            self.species_brains_mean[sid] = means
            self.species_brains_cv[sid] = cvs
                    
    '''def update_species_dendrogram_data(self, world: World):
        distance_matrix = world.gc.distance_matrix
        condensed = distance_matrix[np.triu_indices(len(distance_matrix), k=1)]
        self.species_dendrogram_data = linkage(condensed, method='average')
        self.speciation_cutoff = world.gc.speciation_cutoff'''
