from multiprocessing import Queue
from typing import Dict, List
from world import World
import numpy as np
import time


class Metrics():
    def __init__(self, step_freq=60) -> None:
        self.queue = Queue()

        self.population = []
        self.births = []
        self.deaths = []
        
        self.genes_mean_values = {}
        self.genes_std_values = {}

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
        self.queue.put(self.snapshot())
        
    def update_frame_per_s(self):
        cur_time = time.time_ns()
        self.step_per_s = self.step_freq * 1e9/(cur_time - self.prev_time)
        self.prev_time = cur_time

    def update_attr_metrics(self, world: World):     
        for k in world.genetic_context.genes_mean.keys():
            if f"mean_{k}" not in self.genes_mean_values:
                self.genes_mean_values[f"mean_{k}"] = [world.genetic_context.genes_mean[k]]
                self.genes_std_values[f"std_{k}"] = [world.genetic_context.genes_std[k]]
            else:
                self.genes_mean_values[f"mean_{k}"].append(world.genetic_context.genes_mean[k])
                self.genes_std_values[f"std_{k}"].append(world.genetic_context.genes_std[k])


    def snapshot(self):
        return {
            "n_step": self.n_step,
            "step_per_s": self.step_per_s,
            "population": self.population[:],
            "births": self.births[:],
            "deaths": self.deaths[:],
            **self.genes_mean_values,
            **self.genes_std_values# unpack copy of dict
        }