from typing import TYPE_CHECKING, Dict, List, Tuple
from agent.agent import Agent
import numpy as np
import utils.pos_utils as posUtils
from foodMap import FoodMap
from geneticContext import GeneticContext
from agent.brain import Brain


class World():
    width:int
    height:int
    agents:List[Agent]

    def __init__(self, world_config, agent_config) -> None:
        self.world_config = world_config
        self.agent_config = agent_config
        self.width = world_config['width']
        self.height = world_config['height']
        posUtils.init(self.width, self.height)
        # food
        self.food_map = FoodMap(self.width, self.height, base_energy= world_config['food_energy'],
                                base_size= world_config['food_size'], 
                                rate_per_pixel= world_config['food_spawn_rate_per_pixel'])
        self.food_map.generate_biomes()
        self.food_map.step(world_config['food_initial_n_step'])
        # genetic context
        self.genetic_context = GeneticContext()
        # agents
        self.agents = []
        # metrics
        self.step_count = 0
        self.n_agents = 0
        self.n_births = 0
        self.n_deaths = 0
        self.create(n_agent=world_config['n_agent'])

    def create(self, n_agent=10):
        for _ in range(n_agent):
            agent = Agent.from_config(self.agent_config)
            self.add_agent(agent)
    
    def add_agent(self, agent:Agent):
        self.n_agents += 1
        self.n_births += 1
        self.agents.append(agent)

    def delete_agent(self, agent:Agent):
        self.n_agents -= 1
        self.n_deaths += 1
        self.agents.remove(agent)
    
    def add_food(self, pos=None, energy=None):
        energy = self.food_map.base_energy if energy is None else energy
        if energy == 0:
            return
        x = pos[0] if pos is not None else np.random.randint(0, self.width)
        y = pos[1] if pos is not None else np.random.randint(0, self.height)
        self.food_map.add_food((x, y), energy)
    
    def delete_food(self, pos):
        self.food_map.delete_food(pos)
    
    def get_food_energy(self, pos):
        return self.food_map.get_food_energy(pos)
    
    def get_nearest_food(self, pos, radius: int) -> Tuple[posUtils.Pos|None, float]:
        return self.food_map.get_nearest_food(pos, radius)
    
    def get_nearest_agent(self, from_agent: Agent, radius: int, cond=None) -> Tuple[Agent|None, float]:
        # TODO add condition for nearest agent (ie ready_to_mate or attackable)
        if len(self.agents) <= 1:
            return None, 0
        neareast = min(
            ((posUtils.distance(from_agent.get_pos(), agt.get_pos()), agt) for agt in self.agents if agt != from_agent),
            key=lambda x: x[0]
        )
        dist, agent = neareast
        return (agent, dist) if dist <= radius else (None, 0)
    
    def update_agents(self):
        for agt in self.agents[:]:
            agt.step(self) 
        # delete dead agents
        dead_agents = [agt for agt in self.agents if not agt.alive]
        for agt in dead_agents:
            self.add_food(agt.get_pos(), agt.energy)
            self.delete_agent(agt)  
            
    def update_food(self): 
        if self.step_count % self.world_config['food_update_step_freq'] == 0 :
            self.food_map.step()
            
    def update_genetic_context(self):
        self.genetic_context.update_stats([a.dna for a in self.agents])
        if self.step_count %50 == 0:
            self.genetic_context.compute_species(self.agents)

    def step(self):
        self.step_count += 1
        self.update_genetic_context()
        self.update_agents()
        self.update_food()
        
        
        