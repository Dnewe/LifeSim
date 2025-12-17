import utils.pos_utils as posUtils
from agent.dna import DNA
from typing import TYPE_CHECKING, Self
if TYPE_CHECKING:
    from world.world import World
from agent.goal import REPRODUCE_IDLE_TIME, IdleGoal, Goal
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
    species: int
    age:int
    # actions
    ready_to_reproduce:bool
    goal:Goal
    idle_time: int
    wander_pos:posUtils.Pos|None
   
    
    @classmethod
    def from_config(cls, config):
        dna = DNA.from_config(config['dna'])
        dna.mutate(config['initial_mutation_scale'])
        brain = Brain.from_config(config['brain'])
        energy = dna.max_energy*0.75
        rdm_age = np.random.randint(0, int(dna.lifespan*0.5))
        return cls(dna, brain, pos=posUtils.random_pos(), gen=0, energy=energy, age=rdm_age)
    
    @classmethod
    def clone(cls, other:Self, mutate = True):
        dna = DNA.clone(other.dna)
        if mutate: dna.mutate()
        dna.compute_attributes()
        brain = Brain.clone(other.brain)
        pos = other.get_pos()
        gen = other.generation + 1
        species = -1 # other.species
        energy = other.dna.energy_to_reproduce
        return cls(dna, brain, pos, gen, energy=energy, species=species)

    @classmethod
    def from_parents(cls, agent1:Self, agent2:Self, mutate = True):
        dna = DNA.from_parents(agent1.dna, agent2.dna)
        if mutate: dna.mutate()
        dna.compute_attributes()
        brain = Brain.from_brain1_brain2(agent1.brain, agent2.brain)
        pos = posUtils.midpoint(agent1.get_pos(), agent2.get_pos())
        gen = max(agent1.generation, agent2.generation)+1
        species = np.random.choice([agent1.species, agent2.species])
        energy = agent1.dna.energy_to_reproduce + agent2.dna.energy_to_reproduce
        return cls(dna, brain, pos, gen, energy=energy, species=species)


    def __init__(self, dna:DNA, brain:'Brain', pos: posUtils.Pos, gen:int, energy:float, age:float=0, species:int=-1) -> None:
        self.x, self.y = pos
        self.brain = brain
        self.dna = dna
        self.generation = gen
        self.species = species
        self._init_vars(energy, age)

    def _init_vars(self, energy, age):
        self.age = age
        self.health = self.dna.max_health
        self.alive = True
        self.energy = min(self.dna.max_energy, energy)
        self.satiety = 0
        self.goal = IdleGoal()
        self.wander_pos = None
        self.idle_time = 25
        self.ready_to_reproduce = False
        self.reproduce_cooldown = self.dna.reproduce_cooldown - age

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
        # clip values
        self.energy = min(self.energy, self.dna.max_energy)
        self.satiety = min(self.satiety, self.dna.max_satiety)
        # alive / dead
        if self.energy <= 0 or self.age >= self.dna.lifespan or self.health <= 0:
            print(f'died at: lifespan-age={int(self.dna.lifespan - self.age)}, energy={int(self.energy)}, health={int(self.health)}')
            self.die()
        # attributes
        if self.satiety > 0:
            self.satiety -= self.goal.cost
        else:
            self.energy -= self.goal.cost
        self.health = min(self.dna.max_health, self.health + self.dna.regeneration)
        # timer
        self.age += 1
        self.idle_time = max(0, self.idle_time-1)
        self.reproduce_cooldown = max(0, self.reproduce_cooldown-1)
        # reproduction
        if self.energy >= self.dna.energy_to_reproduce and self.age >= self.dna.maturity_age and self.reproduce_cooldown==0:
            self.ready_to_reproduce = True
        elif self.energy < self.dna.energy_to_reproduce:
            self.ready_to_reproduce = False
        # update cells
        new_cx, new_cy = posUtils.grid_pos(self.get_pos())
        if new_cx == self.cell_x and new_cy == self.cell_y:
            return
        world.grid[self.cell_x][self.cell_y].remove(self)
        self.cell_x = new_cx
        self.cell_y = new_cy
        world.grid[self.cell_x][self.cell_y].append(self)
            
    def reproduce(self):
        """
        called when the agent reproduced.
        Reset reproduce variables and apply energy cost
        """
        self.ready_to_reproduce = False
        self.reproduce_cooldown = self.dna.reproduce_cooldown
        self.idle_time = REPRODUCE_IDLE_TIME
        self.energy -= self.dna.energy_to_reproduce

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
        elif mode == 'gene_values':
            FACTOR = 64
            r = FACTOR * self.dna.gene_values['morphology']**2
            g = FACTOR * self.dna.gene_values['physiology']**2
            b = FACTOR * self.dna.gene_values['sensorial']**2
        elif mode == 'specie':
            cmap = plt.get_cmap('Set2')
            r, g, b = [255*c for c in cmap(self.species%cmap.N)[:3]]
        else: # default
            r, g, b = 0, 255, 255

        r = max(0, min(r, 255))
        g = max(0, min(g, 255))
        b = max(0, min(b, 255))
        return r, g, b
