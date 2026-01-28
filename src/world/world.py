import math
from typing import List, Tuple
import utils.pos_utils as posUtils
from agent.agent import Agent
from world.foodMap import FoodMap
from geneticContext import GeneticContext
from speciator import Speciator
import utils.timeperf as timeperf


class World():
    paused:bool
    width:int
    height:int
    agents:List[Agent]
    grid:List[List[List[Agent]]]
    
    def __init__(self, w, h, cell_size, n_agents, n_food_spawns, speciation_config, worlgen_config, food_config, agent_config) -> None:
        # params
        self.paused = False
        self.width = w
        self.height = h
        self.agent_config = agent_config
        # foodmap
        self.foodmap = FoodMap.from_configs(worlgen_config, food_config)
        self.foodmap.spawn_food(n_food_spawns)
        self.food_spawn_freq = food_config['spawn_freq']
        # agents
        self.gc = GeneticContext.from_config(agent_config)
        self.speciator = Speciator.from_config(self.gc, speciation_config)
        self.agents = []
        # initalization
        self._init_grid(cell_size)
        self._init_counters()
        self._init_agents(n_agent=n_agents)
        self.gc.update(self, global_only=True)
        self.speciator.assign_species(self, True)
    
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
        self.n_actions_per_type = {a: 0 for a in self.agent_config['brain']['utility_scores'].keys()}

    def _init_agents(self, n_agent=10):
        for _ in range(n_agent):
            agent = Agent.from_config(self.agent_config)
            self.add_agent(agent)
    
    ### Events
    
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
    
    def take_food_energy(self, pos, amount = 5):
        energy = self.foodmap.get_food_energy(pos)
        if energy - amount <=0:
            self.foodmap.del_food(pos)
        else:
            self.foodmap.food_arr[pos] = energy - amount
            
    ### Getters
    
    def get_nearest_food(self, pos, radius: int) -> Tuple[posUtils.Pos|None, float]:
        return self.foodmap.get_nearest_food(pos, radius)
    
    @timeperf.timed()
    def get_near_agents(self, from_agent: Agent, radius: int) -> List[Tuple[Agent, float]]:
        '''
        Get agents in radius around an agent,
        returns a list containing agents ordered by distance.
        '''
        if len(self.agents) <= 1:
            return []
        
        cx, cy = posUtils.grid_pos(from_agent.get_pos())
        cells_offset = math.ceil(radius / self.cell_size)
        agents_dists = []
        for dx in range(-cells_offset, cells_offset+1):
            for dy in range(-cells_offset, cells_offset+1):
                nx = cx + dx
                ny = cy + dy
                if 0 <= nx < self.grid_w and 0 <= ny < self.grid_h:
                    for other_agent in self.grid[nx][ny]:
                        if other_agent is from_agent:
                            continue
                        dist = posUtils.distance(from_agent.get_pos(), other_agent.get_pos())
                        agents_dists.append((other_agent, dist))
        # sort
        agents_dists.sort(key= lambda a: a[1])
        sorted_agents_dists = [(a,d) for a,d in agents_dists if d<=radius]
        return sorted_agents_dists
    
    ### Updates
    
    def update_grid_pos(self, agent: Agent):
        new_cx, new_cy = posUtils.grid_pos(agent.get_pos())
        if new_cx == agent.cell_x and new_cy == agent.cell_y:
            return
        self.grid[new_cx][new_cy].append(agent)
        self.grid[agent.cell_x][agent.cell_y].remove(agent)
        agent.cell_x = new_cx
        agent.cell_y = new_cy
        
    def exec_event(self, event):
        etype, args = event
        if etype == 'add_agent':
            self.add_agent(**args)
        elif etype == 'eat':
            self.take_food_energy(**args)
        
    def step_agents(self):
        for a in self.agents:
            a.sense(self)
            a.decide()
        for a in self.agents:
            event = a.act()
            if event is not None:
                self.exec_event(event)
        dead_agents = [a for a in self.agents if not a.alive]
        for a in dead_agents:
            self.foodmap.add_food(a.get_pos(), a.energy)
            self.delete_agent(a) 
        for a in self.agents:
            self.update_grid_pos(a)                 
        
    def update_counter(self):
        self.step_count += 1
        counts = {}
        for a in self.agents:
            action = a.last_action
            self.n_actions_per_type[action] += 1
            sid = a.species
            counts[sid] = counts.get(sid, 0) + 1
        self.n_agents_per_species = counts

    @timeperf.timed()
    def step(self):
        if self.paused:
            return
        self.step_agents()
        self.update_counter()
        self.speciator.update(self)
        self.gc.update(self)
        self.foodmap.update(self)
        
        
        