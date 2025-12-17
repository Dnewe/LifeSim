from typing import TYPE_CHECKING, Dict, List, Tuple
from agent.agent import Agent
import numpy as np
import utils.pos_utils as posUtils
from world.biomeMap import BiomeMap
from geneticContext import GeneticContext
from agent.brain import Brain
import math


class World():
    width:int
    height:int
    agents:List[Agent]
    
    @classmethod
    def from_config(cls, world_config, agent_config):
        return cls(world_config['worldgen']['width'], 
                   world_config['worldgen']['height'], 
                   world_config['worldgen']['grid_cell_size'], 
                   world_config['n_agents'], 
                   world_config['n_food_spawns'],
                   world_config['speciation'],
                   world_config['worldgen'],
                   world_config['food'],
                   agent_config,
                   )
    
    def __init__(self, w, h, cell_size, n_agents, n_food_spawns, speciation_config, worlgen_config, food_config, agent_config) -> None:
        # params
        self.width = w
        self.height = h
        self.agent_config = agent_config
        # foodmap
        self.foodmap = BiomeMap.from_configs(worlgen_config, food_config)
        self.foodmap.spawn_food(n_food_spawns)
        self.food_spawn_freq = food_config['spawn_freq']
        # agents
        self.gc = GeneticContext.from_config(speciation_config)
        self.agents = []
        # initalization
        self._init_grid(cell_size)
        self._init_counters()
        self._init_agents(n_agent=n_agents)
        self.gc.update_stats(self.agents, global_only=True)
        self.gc.assign_species(self.agents)
        
    def _init_grid(self, cell_size):
        self.cell_size = cell_size
        self.grid_w = posUtils.GRID_W
        self.grid_h = posUtils.GRID_H
        self.grid = [[[] for _ in range(self.grid_h)] for _ in range(self.grid_w)]
        
    def _init_counters(self):
        self.step_count = 0
        self.n_agents = 0
        self.n_agents_per_species = {}
        self.n_births = 0
        self.n_deaths_per_type = {"age": 0, "starvation": 0, "killed": 0}

    def _init_agents(self, n_agent=10):
        for _ in range(n_agent):
            agent = Agent.from_config(self.agent_config)
            self.add_agent(agent)
    
    def add_agent(self, agent:Agent):
        self.n_agents += 1
        self.n_births += 1
        cx, cy = posUtils.grid_pos(agent.get_pos())
        self.grid[cx][cy].append(agent)
        self.agents.append(agent)
        agent.cell_x = cx
        agent.cell_y = cy

    def delete_agent(self, agent:Agent):
        self.n_agents -= 1
        self.n_deaths_per_type[agent.death_reason] += 1
        self.agents.remove(agent)
        self.grid[agent.cell_x][agent.cell_y].remove(agent)
    
    def add_food(self, pos, energy):
        if energy<=0:
            return
        self.foodmap.add_food(pos, energy)
    
    def delete_food(self, pos):
        self.foodmap.del_food(pos)
    
    def take_food_energy(self, pos, amount = 5):
        energy = self.foodmap.get_food_energy(pos)
        amount = min(energy, amount)
        if energy - amount <=0:
            self.foodmap.del_food(pos)
        else:
            self.foodmap.food_arr[pos] = energy - amount
        return amount
    
    def get_nearest_food(self, pos, radius: int) -> Tuple[posUtils.Pos|None, float]:
        return self.foodmap.get_nearest_food(pos, radius)
    
    def get_nearest_agent(self, from_agent: Agent, radius: int, cond=None) -> Tuple[Agent|None, float]:
        if len(self.agents) <= 1:
            return None, 0
        closest = None
        closest_dist = float("inf")
        
        cx, cy = posUtils.grid_pos(from_agent.get_pos())
        cells_offset = math.ceil(radius / self.cell_size)
        for dx in range(-cells_offset, cells_offset+1):
            for dy in range(-cells_offset, cells_offset+1):
                nx = cx + dx
                ny = cy + dy
                if 0 <= nx < self.grid_w and 0 <= ny < self.grid_h:
                    for other_agent in self.grid[nx][ny]:
                        if other_agent is from_agent:
                            continue
                        dist = posUtils.distance(from_agent.get_pos(), other_agent.get_pos())
                        if dist < closest_dist:
                            closest = other_agent
                            closest_dist = dist
                        
        return (closest, closest_dist) if closest_dist <= radius else (None, 0)
    
    def update_agents(self):
        for agt in self.agents[:]:
            agt.step(self) 
        # delete dead agents
        dead_agents = [agt for agt in self.agents if not agt.alive]
        for agt in dead_agents:
            self.add_food(agt.get_pos(), agt.energy)
            self.delete_agent(agt)  
            
    def update_food(self): 
        if self.step_count % self.food_spawn_freq == 0 :
            self.foodmap.spawn_food()
            
    def update_genetic_context(self):
        self.gc.update(self)
        
    def update_counter(self):
        self.step_count += 1
        counts = {}
        for a in self.agents:
            sid = a.species
            counts[sid] = counts.get(sid, 0) + 1
        self.n_agents_per_species = counts

    def step(self):
        self.update_agents()
        self.update_genetic_context()
        self.update_food()
        self.update_counter()
        
        
        