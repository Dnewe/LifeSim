from typing import Dict, List
from multiprocessing import Queue
from world import World
from sklearn.decomposition import PCA
import numpy as np
import time


class Metrics():
    def __init__(self, step_freq=60) -> None:
        self.queue = Queue()

        self.population = []
        self.births = []
        self.deaths = []
        
        self.genes_mean_values = {}
        self.genes_cv_values = {}
        self.species_scatter_points = np.zeros((2,2))
        self.species_labels = []

        self.step_freq = step_freq
        self.n_step = 0
        self.step_per_s = 0
        self.prev_time = 0

    def update(self, world: World):
        self.n_step = world.step_count
        if self.n_step % self.step_freq != 0:
            return
        self.update_frame_per_s()
        self.population.append(world.n_agents)
        self.births.append(world.n_births)
        self.deaths.append(world.n_deaths)
        self.update_attr_metrics(world)
        self.update_species_scatter(world)
        self.queue.put(self.snapshot())
        
    def update_frame_per_s(self):
        cur_time = time.time_ns()
        self.step_per_s = self.step_freq * 1e9/(cur_time - self.prev_time)
        self.prev_time = cur_time

    def update_attr_metrics(self, world: World):     
        for k in world.genetic_context.genes_mean.keys():
            if f"mean_{k}" not in self.genes_mean_values:
                self.genes_mean_values[f"mean_{k}"] = [world.genetic_context.genes_mean[k]]
                self.genes_cv_values[f"cv_{k}"] = [world.genetic_context.genes_std[k]]
            else:
                self.genes_mean_values[f"mean_{k}"].append(world.genetic_context.genes_mean[k])
                self.genes_cv_values[f"cv_{k}"].append(world.genetic_context.genes_std[k])

    def update_species_scatter(self, world: World):
        dnas = [a.dna for a in world.agents]
        genes = dnas[0].gene_values.keys()
        mat = np.array([[dna.gene_values[g] for g in genes] for dna in dnas])
        pca = PCA(n_components=2)
        self.species_labels = np.array([a.specie for a in world.agents])
        self.species_scatter_points = pca.fit_transform(mat)

    def snapshot(self):
        return {
            "n_step": self.n_step,
            "step_per_s": self.step_per_s,
            "population": self.population[:],
            "births": self.births[:],
            "deaths": self.deaths[:],
            "species_scatter_points_and_labels": (self.species_scatter_points, self.species_labels),
            **self.genes_mean_values,
            **self.genes_cv_values# unpack copy of dict
        }