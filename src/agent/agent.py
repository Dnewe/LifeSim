import numpy as np
import matplotlib.pyplot as plt
import utils.pos_utils as posUtils
from typing import TYPE_CHECKING, Self
if TYPE_CHECKING:
    from world.world import World
from agent.goal import REPRODUCE_IDLE_TIME, IdleGoal, Goal, GOALS_MAP
from agent.genome import Genome
from agent.brain.brain import Brain
import utils.timeperf as timeperf

# proportion of agent energy spawned as food upon death
DEATH_ENERGY_PROP = 1.


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
    can_reproduce:bool
    goal:Goal
    idle_time: int
    wander_pos:posUtils.Pos|None
   
    
    @classmethod
    def from_config(cls, config):
        genome = Genome.from_config(config['genome'])
        genome.mutate(config['initial_genome_mutation_factor'], n_mutations= -1)
        brain = Brain.from_config(config['brain'])
        brain.mutate(config['initial_brain_mutation_factor'], n_mutations= -1)
        energy = genome.max_energy * 0.75
        rdm_age = np.random.randint(0, int(genome.max_age*0.5))
        return cls(genome, brain, pos=posUtils.random_pos(), gen=0, energy=energy, age=rdm_age)
    
    @classmethod
    def clone(cls, other:Self, mutate = True):
        genome = Genome.clone(other.genome)
        if mutate: genome.mutate()
        genome.compute_attributes()
        brain = Brain.clone(other.brain)
        if mutate: brain.mutate()
        pos = other.get_pos()
        gen = other.generation + 1
        species = -1 # unassigned species
        energy = other.genome.energy_to_reproduce
        return cls(genome, brain, pos, gen, energy=energy, species=species)

    @classmethod
    def from_parents(cls, agent1:Self, agent2:Self, mutate = True):
        genome = Genome.from_parents(agent1.genome, agent2.genome)
        if mutate: genome.mutate()
        genome.compute_attributes()
        brain = Brain.from_brain1_brain2(agent1.brain, agent2.brain)
        pos = posUtils.midpoint(agent1.get_pos(), agent2.get_pos())
        gen = max(agent1.generation, agent2.generation)+1
        species = -1 # unassigned species 
        energy = agent1.genome.energy_to_reproduce + agent2.genome.energy_to_reproduce
        return cls(genome, brain, pos, gen, energy=energy, species=species)


    def __init__(self, genome:Genome, brain:'Brain', pos: posUtils.Pos, gen:int, energy:float, age:int=0, species:int=-1) -> None:
        self.x, self.y = pos
        self.brain = brain
        self.genome = genome
        self.generation = gen
        self.species = species
        # variables
        self.age = age
        self.health = self.genome.max_health
        self.alive = True
        self.energy = min(self.genome.max_energy, energy)
        self.goal = IdleGoal()
        self.last_action = 'idle'
        #self.last_action = 'idle'
        self.wander_pos = None
        self.idle_time = 0
        self.can_reproduce = False
        self.reproduce_cooldown = self.genome.reproduce_cooldown - age

    def sense(self, world: 'World'):
        self.brain.sense(self, world)

    @timeperf.timed()
    def decide(self):
        action, args = self.brain.decide() if self.idle_time<=0 else ('idle', None)
        self.goal = GOALS_MAP[action](args)
        self.last_action = action

    @timeperf.timed()
    def act(self):
        event = self.goal.exec(self)
        self.update()
        return event

    def update(self):
        # clip values
        self.energy = min(self.energy, self.genome.max_energy)
        # alive / dead
        if self.energy <= 0 or self.age >= self.genome.max_age or self.health <= 0:
            print(f'died at: max_age-age={int(self.genome.max_age - self.age)}, energy={int(self.energy)}, health={int(self.health)}')
            self.die()
        # decay energy
        self.energy -= self.goal.cost
        # regeneration
        self.health = min(self.genome.max_health, self.health + self.genome.regeneration)
        # timer
        self.age += 1
        self.idle_time = max(0, self.idle_time-1)
        self.reproduce_cooldown = max(0, self.reproduce_cooldown-1)
        # reproduction
        if self.energy >= self.genome.energy_to_reproduce and self.age >= self.genome.maturity_age and self.reproduce_cooldown==0:
            self.can_reproduce = True
        elif self.energy < self.genome.energy_to_reproduce:
            self.can_reproduce = False
            
    def reproduce(self):
        """
        called when the agent reproduced.
        Reset reproduce variables and apply energy cost
        """
        self.can_reproduce = False
        self.reproduce_cooldown = self.genome.reproduce_cooldown
        self.idle_time = REPRODUCE_IDLE_TIME
        self.energy -= self.genome.energy_to_reproduce

    def die(self):   
        self.death_reason = "starvation" if self.energy <= 0 else "killed" if self.health <= 0 else "age"
        self.energy *= DEATH_ENERGY_PROP
        self.alive = False
        
    def walk(self, dest_pos, speed_factor=1.) -> bool:
        """
        walks towards destination position.
        returns whether destination reached or not.
        """
        self.speed_factor = speed_factor
        step_size = self.genome.speed * self.speed_factor
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
        return int(self.genome.size)
    
    def get_color(self, mode='default'): 
        if mode == 'generation':
            r = 0
            g = max(0, min(255, (510 - 20*(self.generation))%511))
            b = max(0, min(255, (0 + 20*(self.generation))%511))
        elif mode == 'gene_values':
            FACTOR = 64
            r = FACTOR * self.genome.gene_values['morphology']**2
            g = FACTOR * self.genome.gene_values['physiology']**2
            b = FACTOR * self.genome.gene_values['sensorial']**2
        elif mode == 'specie':
            cmap = plt.get_cmap('Set2')
            r, g, b = [255*c for c in cmap(self.species%cmap.N)[:3]]
        else: # default
            r, g, b = 0, 255, 255

        r = max(0, min(r, 255))
        g = max(0, min(g, 255))
        b = max(0, min(b, 255))
        return r, g, b
