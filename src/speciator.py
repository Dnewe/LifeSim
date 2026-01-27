import numpy as np

from typing import List, TYPE_CHECKING
if TYPE_CHECKING:
    from agent.agent import Agent
    from world.world import World
    from geneticContext import GeneticContext
from agent.genome import Genome
from agent.brain.brain import Brain
from geneticContext import GeneticContext
import utils.timeperf as timeperf


class Species:
    def __init__(self, representative: 'Agent', _id) -> None:
        self.id = _id
        self.representative = representative
        self.members: List['Agent'] = []


class Speciator:
    gc: GeneticContext
    def __init__(self, gc, speciation_cutoff= 4.0, genome_brain_ratio=0.5, alpha= 1.0, reassign_species=True, update_freq=100) -> None:
        self.gc = gc
        self.speciation_cutoff = speciation_cutoff
        self.genome_brain_ratio = genome_brain_ratio
        self.alpha = alpha
        self.reassign_species = reassign_species
        self.update_freq = update_freq
        # species
        self.species:List[Species] = []
        self.n_species = 0
        self.next_species_id = 0
        
    @classmethod
    def from_config(cls, gc: GeneticContext, config):
        return cls(gc, config['speciation_cutoff'], config['genome_brain_ratio'], config['alpha'], config['reassign_species'], config['representative_update_freq'])
    
    @timeperf.timed()
    def update(self, world: 'World'):
        if world.step_count % self.update_freq == 0:
            self.assign_species(world, self.reassign_species)
            self.refresh_representatives()
            self.n_species = len(self.species)
    
    def assign_species(self, world: 'World', reassign=True):
        for s in self.species:
            s.members.clear()
        for a in world.agents:
            if not reassign and a.species in [s.id for s in self.species]: # only find species for agent without species
                for s in self.species:
                    if s.id == a.species: s.members.append(a)
                continue
            closest_species = None
            min_distance = float("inf")
            for species in self.species:
                d = self.agent_distance(a, species.representative, self.gc) # Genome.distance(a.genome, species.representative.genome, self.gc.genes_std, alpha=self.alpha) + Brain.distance(a.brain)
                if d < min_distance:
                    min_distance = d
                    closest_species = species
            if closest_species is not None and min_distance < self.speciation_cutoff:
                a.species = closest_species.id
                closest_species.members.append(a)
            else:
                # new species
                s = Species(a, self.next_species_id)
                self.next_species_id += 1
                s.members.append(a)
                self.species.append(s)
                a.species = s.id
                print(f'New species {s.id}')
        self.species = [s for s in self.species if s.members]
    
    def refresh_representatives(self):
        for s in self.species:
            if len(s.members) <= 2:
                s.representative = s.members[0]
            else:
                s.representative = self.choose_medoid(s) 
                
    def choose_medoid(self, species: 'Species'):
        members = species.members
        best_agent = members[0]
        best_score = float("inf")
        for a in members:
            score = 0.0
            for b in members:
                if a is b:
                    continue
                score += self.agent_distance(a, species.representative, self.gc) # Genome.distance(a.genome, b.genome, self.gc.genes_std, alpha=self.alpha)
            if score < best_score:
                best_score = score
                best_agent = a
        return best_agent
    
    def agent_distance(self, agent1:'Agent', agent2:'Agent', gc:'GeneticContext'):
        d_genome = Genome.distance(agent1.genome, agent2.genome, gc.genes_std, alpha=self.alpha)
        d_brain = Brain.distance(agent1.brain, agent2.brain, gc.brains_std, alpha=self.alpha)
        return d_genome * self.genome_brain_ratio + d_brain * (1-self.genome_brain_ratio)
        
        
        