import numpy as np

from typing import List, TYPE_CHECKING
if TYPE_CHECKING:
    from agent.genome import Genome
    from agent.agent import Agent
    from world.world import World
from geneticContext import GeneticContext


class Species:
    def __init__(self, representative_genome: 'Genome', _id) -> None:
        self.id = _id
        self.genome = representative_genome
        self.members: List['Agent'] = []


class Speciator:
    gc: GeneticContext
    def __init__(self, gc, speciation_cutoff= 4.0, alpha= 1.0, reassign_species=True, representative_update_freq=100) -> None:
        self.gc = gc
        self.speciation_cutoff = speciation_cutoff
        self.alpha = alpha
        self.reassign_species = reassign_species
        self.representative_update_freq = representative_update_freq
        # species
        self.species:List[Species] = []
        self.n_species = 0
        self.next_species_id = 0
        
    @classmethod
    def from_config(cls, gc: GeneticContext, config):
        return cls(gc, config['speciation_cutoff'], config['alpha'], config['reassign_species'], config['representative_update_freq'])
        
    def update(self, world: 'World'):
        self.assign_species(world, self.reassign_species)
        if world.step_count % self.representative_update_freq == 0:
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
                d = self.genome_distance(a.genome, species.genome)
                if d < min_distance:
                    min_distance = d
                    closest_species = species
            if closest_species is not None and min_distance < self.speciation_cutoff:
                a.species = closest_species.id
                closest_species.members.append(a)
            else:
                # new species
                s = Species(a.genome, self.next_species_id)
                self.next_species_id += 1
                s.members.append(a)
                self.species.append(s)
                a.species = s.id
        self.species = [s for s in self.species if s.members]
    
    def refresh_representatives(self):
        for s in self.species:
            if len(s.members) <= 2:
                s.genome = s.members[0].genome
            else:
                s.genome = self.choose_medoid(s)
                
    def genome_distance(self, genome1: 'Genome', genome2: 'Genome', eps=1e-9):
        v1, keys = self.genome_vector(genome1)
        v2, _    = self.genome_vector(genome2)
        stds = self.gc.genes_std
        weights = np.array([1.0 for k in keys]) # TODO weights for genes
        # normalized euclidean
        diff = (v1 - v2) / (stds + eps)
        euclid = np.sqrt(np.sum(weights * diff * diff))
        # cosine distance (directional)
        cosine = 1.0 - np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + eps)
        return euclid * (1.0 + self.alpha * cosine)   
                
    def choose_medoid(self, species: 'Species'):
        members = species.members
        best_genome = members[0].genome
        best_score = float("inf")
        for a in members:
            score = 0.0
            for b in members:
                if a is b:
                    continue
                score += self.genome_distance(a.genome, b.genome)
            if score < best_score:
                best_score = score
                best_genome = a.genome
        return best_genome
    
    def genome_vector(self, genome: 'Genome'):
        keys = sorted(genome.gene_values.keys())
        return np.array([np.log1p(genome.gene_values[k]) for k in keys]), keys
        