import utils.pos_utils as posUtils
from agent.dna import DNA
from typing import TYPE_CHECKING, Self
if TYPE_CHECKING:
    from world import World
from agent.goal import MATING_IDLE_TIME, IdleGoal, Goal
import numpy as np
from agent.brain import Brain
import matplotlib.pyplot as plt


class Agent():
    # pos
    x:float
    y:float
    cell_x:int
    cell_y:int
    # attributes
    alive:bool
    health: float
    energy:float
    generation:int
    specie: int
    age:int
    # actions
    ready_to_mate:bool
    goal:Goal
    idle_time: int
    wander_pos:posUtils.Pos|None
   
    
    @classmethod
    def from_config(cls, config):
        dna = DNA.from_config(config['dna'])
        dna.mutate(config['initial_mutation_scale'])
        brain = Brain.from_config(config['brain'])
        energy = dna.max_energy*0.8
        rdm_age = np.random.randint(0, int(dna.lifespan*0.5))
        return cls(dna, brain, pos=posUtils.random_pos(), gen=0, energy=energy, age=rdm_age)

    @classmethod
    def from_parents(cls, agent1:Self, agent2:Self, mutate = True):
        # DNA
        dna = DNA.from_parents(agent1.dna, agent2.dna)
        if mutate: dna.mutate()
        dna.compute_attributes()
        # Brain
        brain = Brain.from_brain1_brain2(agent1.brain, agent2.brain)
        # create Agent
        pos = posUtils.midpoint(agent1.get_pos(), agent2.get_pos())
        gen = max(agent1.generation, agent2.generation)+1
        energy = agent1.dna.energy_to_mate + agent2.dna.energy_to_mate
        return cls(dna, brain, pos, gen, energy=energy)


    def __init__(self, dna:DNA, brain:'Brain', pos: posUtils.Pos, gen:int, energy:float, age:float=0) -> None:
        self.x, self.y = pos
        self.brain = brain
        self.dna = dna
        self.generation = gen
        self._init_vars(energy, age)

    def _init_vars(self, energy, age):
        self.age = age
        self.specie = 0
        self.health = self.dna.max_health
        self.alive = True
        self.energy = min(self.dna.max_energy, energy)
        self.goal = IdleGoal()
        self.wander_pos = None
        self.idle_time = 25
        self.ready_to_mate = False
        self.mating_cooldown = self.dna.mating_cooldown

    def step(self, world):
        self.sense(world)
        self.decide()
        self.act(world)
        self.update(world)

    def sense(self, world: 'World'):
        self.brain.sense(self, world)

    def decide(self):
        self.goal = self.brain.decide() if self.idle_time<=0 else IdleGoal()

    def act(self, world):
        self.goal.exec(self, world)

    def update(self, world: 'World'):
        # alive / dead
        if self.energy <= 0 or self.age >= self.dna.lifespan or self.health <= 0:
            print(f'died at: lifespan-age={int(self.dna.lifespan - self.age)}, energy={int(self.energy)}, health={int(self.health)}')
            self.die()
        # attributes
        self.energy -= self.goal.cost
        self.health = min(self.dna.max_health, self.health + self.dna.regeneration)
        # timer
        self.age += 1
        self.idle_time = max(0, self.idle_time-1)
        self.mating_cooldown = max(0, self.mating_cooldown-1)
        # reproduction
        if self.energy >= self.dna.energy_to_mate and self.age >= self.dna.maturity_age and self.mating_cooldown==0:
            self.ready_to_mate = True
        elif self.energy < self.dna.energy_to_mate:
            self.ready_to_mate = False
        # update cells
        new_cx, new_cy = posUtils.grid_pos(self.get_pos())
        if new_cx == self.cell_x and new_cy == self.cell_y:
            return
        world.grid[self.cell_x][self.cell_y].remove(self)
        self.cell_x = new_cx
        self.cell_y = new_cy
        world.grid[self.cell_x][self.cell_y].append(self)
            
    def mate(self):
        """
        called when the agent mated.
        Reset mating variables and apply energy cost
        """
        self.ready_to_mate = False
        self.mating_cooldown = self.dna.mating_cooldown
        self.idle_time = MATING_IDLE_TIME
        self.energy -= self.dna.energy_to_mate

    def die(self):   
        self.death_reason = "starvation" if self.energy <= 0 else "killed" if self.health <= 0 else "age"
        self.alive = False
        
    def walk(self, dest_pos, speed_factor=1.) -> bool:
        """
        walks towards destination position.
        returns whether destination reached or not.
        """
        self.speed_factor = speed_factor
        step_size = self.dna.speed * self.speed_factor
        x_dest, y_dest = dest_pos
        dx = x_dest - self.x
        dy = y_dest - self.y
        dist = np.hypot(dx, dy)
        if dist ==0:
            return True
        if dist <= step_size:
            self.x = x_dest
            self.y = y_dest
            return True
        else:
            self.x += step_size * dx / dist
            self.y += step_size * dy / dist
            return False


    def get_pos(self):
        return posUtils.clip_pos((int(self.x), int(self.y)))
    
    def get_size(self):
        return int(self.dna.size)
    
    def get_color(self, mode='default'): 
        if mode == 'generation':
            r = 0
            g = max(0, min(255, (510 - 20*(self.generation))%511))
            b = max(0, min(255, (0 + 20*(self.generation))%511))
        elif mode == 'dna_distance':
            factor = 1. # self.dna.distance_from_default() TODO
            r = 0
            g = 255*(1-factor)
            b = 255*factor
        elif mode == 'gene_values':
            FACTOR = 64
            r = FACTOR * self.dna.gene_values['morphology']**2
            g = FACTOR * self.dna.gene_values['physiology']**2
            b = FACTOR * self.dna.gene_values['sensorial']**2
        elif mode == 'specie':
            cmap = plt.get_cmap('tab20')
            r, g, b = [255*c for c in cmap(self.specie%20)[:3]]
        else: # default
            r, g, b = 0, 255, 255

        r = max(0, min(r, 255))
        g = max(0, min(g, 255))
        b = max(0, min(b, 255))
        return r, g, b
