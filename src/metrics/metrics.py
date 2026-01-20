from typing import Dict, List
from multiprocessing import Queue
from world.world import World
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import linkage
import pandas as pd
import numpy as np
import time


class Metrics():
    def __init__(self, genes, step_freq=60) -> None:
        self.genes = genes
        self.queue = Queue(maxsize=1)
        self.step_freq = step_freq
        self.prev_time = 0
        # sim info
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
        self.genes_means = {g: [] for g in genes}
        self.genes_cvs = {g: [] for g in genes}
        self.species_genes_mean = {}
        self.species_genes_cv = {}
        # brain
        self.actions_density_data = {'count': [], 'action': [], 'step': []}
        self.brains_means = {} # lists of mean in dict of weights in dict of action of brain
        self.brains_cvs = {}
        self.species_brains_mean = {}
        self.species_brains_cv = {}


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
                self.deaths[type] = [n]
            else: 
                self.deaths[type].append(n)
        # species
        self.n_species = max(1, len(world.gc.species))
        self.update_species_genes_data(world)
        self.update_species_scatter_data(world)
        self.update_species_density_data(world)
        # genes
        self.update_genes_metrics(world)
        # brain
        self.update_actions_density_data(world)
        self.update_brain_metrics(world)

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
            "species_density_df": pd.DataFrame({**self.species_density_data}),
            # genes
            "genes_means": {**self.genes_means},
            "genes_cvs": {**self.genes_cvs},
            "species_genes_mean": self.species_genes_mean,
            "species_genes_cv": self.species_genes_cv,
            # brains
            "actions_density_df": pd.DataFrame({**self.actions_density_data}),
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
        dnas_all = [a.dna for a in world.agents]
        dnas_repr = [s.representative for s in world.gc.species]
        genes = dnas_all[0].gene_values.keys()
        mat_all = np.array([[dna.gene_values[g] for g in genes] for dna in dnas_all if dna not in dnas_repr])
        mat_repr = np.array([[dna.gene_values[g] for g in genes] for dna in dnas_repr])
        if mat_all.shape[0] < 2:
            return  # exit if not enough non-representant agents
        pca = PCA(n_components=2)
        points_all = pca.fit_transform(mat_all)
        points_repr = pca.transform(mat_repr)
        labels_all = np.array([int(a.species) for a in world.agents if a.dna not in dnas_repr])
        labels_repr = np.array([s.id for s in world.gc.species])
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
        means = world.gc.genes_mean
        stds = world.gc.genes_std
        for g in self.genes:
            self.genes_means[g].append(means.get(g, 0))
            self.genes_cvs[g].append(stds.get(g, 0)/means.get(g, 1e-8))
        
    def update_species_genes_data(self, world: World):
        self.species_genes_mean = {s.id: [] for s in world.gc.species}
        self.species_genes_cv = {s.id: [] for s in world.gc.species}
        for s in world.gc.species:
            sid = s.id
            means_dict = world.gc.species_genes_mean.get(sid,{})
            stds_dict = world.gc.species_genes_std.get(sid,{})
            means = [means_dict.get(g,0.) for g in self.genes]
            cvs = [stds_dict.get(g,0.) / means_dict.get(g,1e-8) for g in self.genes]
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
        means = world.gc.brains_mean
        stds = world.gc.brains_std
        for a in means:
            for w in means[a]:
                self.brains_means[a] = self.brains_means.get(a, {})
                self.brains_means[a][w] = self.brains_means[a].get(w, [])
                self.brains_means[a][w].append(means[a].get(w, 0))
                self.brains_cvs[a] = self.brains_cvs.get(a, {})
                self.brains_cvs[a][w] = self.brains_cvs[a].get(w, [])
                self.brains_cvs[a][w].append(stds[a].get(w, 0) / (means[a].get(w, 0) + 1e-8))
        from utils.debug import print_brain_vars
        print_brain_vars(means)
            
    def update_species_brains_data(self, world: World):
        self.species_brains_mean = {s.id: {} for s in world.gc.species}
        self.species_brains_cv = {s.id: {} for s in world.gc.species}
        for s in world.gc.species:
            sid = s.id
            means_dict = world.gc.species_brains_mean.get(sid,{})
            stds_dict = world.gc.species_brains_std.get(sid,{})
            means = {}
            cvs = {}
            for a in means_dict:
                means[a] = {w: means_dict[a].get(w, 0.0) for a in means_dict for w in means_dict[a]}
                cvs[a] = {w: stds_dict[a].get(w, 0.0)  / (means_dict[a].get(w, 0.0) + 1e-8)  for a in means_dict for w in means_dict[a]}
            self.species_brains_mean[sid] = means
            self.species_brains_cv[sid] = cvs
                    
    '''def update_species_dendrogram_data(self, world: World):
        distance_matrix = world.gc.distance_matrix
        condensed = distance_matrix[np.triu_indices(len(distance_matrix), k=1)]
        self.species_dendrogram_data = linkage(condensed, method='average')
        self.speciation_cutoff = world.gc.speciation_cutoff'''
