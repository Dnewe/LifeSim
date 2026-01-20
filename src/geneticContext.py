from typing import TYPE_CHECKING, List
if TYPE_CHECKING:
    from agent.dna import DNA
    from agent.agent import Agent
    from world.world import World
import numpy as np


class Species:
    def __init__(self, representative_dna: 'DNA', _id) -> None:
        self.id = _id
        self.representative = representative_dna
        self.members: List['Agent'] = []
        

class GeneticContext:
    def __init__(self, speciation_cutoff= 4.0, alpha= 1.0, reassign_species=True, representative_update_freq=100) -> None:
        self.speciation_cutoff = speciation_cutoff
        self.alpha = alpha
        self.reassign_species = reassign_species
        self.representative_update_freq = representative_update_freq
        self.species:List[Species] = []
        self.next_species_id = 0
        # genes
        self.genes_mean = {}
        self.genes_std = {}
        self.species_genes_mean = {}
        self.species_genes_std = {}
        # brains
        self.brains_mean = {}
        self.brains_std = {}
        self.species_brains_mean = {}
        self.species_brains_std = {}
        
    @classmethod
    def from_config(cls, config):
        return cls(config['speciation_cutoff'], config['alpha'], config['reassign_species'], config['representative_update_freq'])
    
    def update(self, world: 'World'):
        self.assign_species(world.agents, self.reassign_species)
        self.update_gene_stats(world.agents)
        self.update_brain_stats(world.agents)
        if world.step_count % self.representative_update_freq == 0:
            self.refresh_representatives()
    
        
    def assign_species(self, agents: List['Agent'], reassign=True):
        for s in self.species:
            s.members.clear()
        for agent in agents:
            if not reassign and agent.species in [s.id for s in self.species]: # only find species for agent without species
                for s in self.species:
                    if s.id == agent.species: s.members.append(agent)
                continue
            closest_species = None
            min_distance = float("inf")
            for species in self.species:
                d = self.dna_distance(agent.dna, species.representative)
                if d < min_distance:
                    min_distance = d
                    closest_species = species
            if closest_species is not None and min_distance < self.speciation_cutoff:
                agent.species = closest_species.id
                closest_species.members.append(agent)
            else:
                # new species
                s = Species(agent.dna, self.next_species_id)
                self.next_species_id += 1
                s.members.append(agent)
                self.species.append(s)
                agent.species = s.id
        self.species = [s for s in self.species if s.members]
                
    def refresh_representatives(self):
        for s in self.species:
            if len(s.members) <= 2:
                s.representative = s.members[0].dna
            else:
                s.representative = self.choose_medoid(s)
    
    
    def update_gene_stats(self, agents: List['Agent'], gamma_mean=0.4, gamma_std=0.8, eps=1e-9, global_only=False):
        ### accumulators
        # global
        global_sum = {}
        global_sq_sum = {}
        global_count = 0  
        # per specie   
        specie_sum = {}
        specie_sq_sum = {}
        specie_count = {}
        
        ### accumulate
        for a in agents:
            global_count += 1
            sid = a.species
            if not global_only:
                specie_count[sid] = specie_count.get(sid, 0) + 1
                if sid not in specie_sum:
                    specie_sum[sid] = {}
                    specie_sq_sum[sid] = {}
            for k, v in a.dna.gene_values.items():
                # global
                global_sum[k] = global_sum.get(k, 0.0) + v
                global_sq_sum[k] = global_sq_sum.get(k, 0.0) + v * v
                # per-species
                if not global_only:
                    specie_sum[sid][k] = specie_sum[sid].get(k, 0.0) + v
                    specie_sq_sum[sid][k] = specie_sq_sum[sid].get(k, 0.0) + v * v 
                
        ### compute stats    
        # global        
        for k in global_sum:
            mean = global_sum[k] / global_count
            var = global_sq_sum[k] / global_count - mean * mean
            std = (var ** 0.5) + eps
            self.genes_mean[k] = gamma_mean * self.genes_mean.get(k, mean) + (1-gamma_mean) * mean
            self.genes_std[k] = gamma_std * self.genes_std.get(k, std) + (1-gamma_std) * std
        
        if global_only:
            return
        # per specie
        self.species_genes_mean = {}
        self.species_genes_std = {}
        for sid in specie_count:
            self.species_genes_mean[sid] = {}
            self.species_genes_std[sid] = {}
            cnt = specie_count[sid]
            for k in global_sum:
                if cnt == 0 or k not in specie_sum[sid]: # if empty specie
                    self.species_genes_mean[sid][k] = eps
                    self.species_genes_std[sid][k]  = eps
                    continue
                m = specie_sum[sid][k] / cnt
                v = specie_sq_sum[sid][k] / cnt - m * m
                sd = (v ** 0.5) + eps
                self.species_genes_mean[sid][k] = m
                self.species_genes_std[sid][k]  = sd

    def update_brain_stats(self, agents: List['Agent'], gamma_mean=0.4, gamma_std=0.8, eps=1e-9, global_only=False):
        ### accumulators
        # global
        global_sum = {}
        global_sq_sum = {}
        global_count = 0  
        # per specie   
        specie_sum = {}
        specie_sq_sum = {}
        specie_count = {}
        
        ### accumulate
        for a in agents:
            global_count += 1
            sid = a.species
            if not global_only:
                specie_count[sid] = specie_count.get(sid, 0) + 1
                if sid not in specie_sum:
                    specie_sum[sid] = {}
                    specie_sq_sum[sid] = {}
            for a, weights in a.brain.get_data().items():
                global_sum[a] = global_sum.get(a, {})
                global_sq_sum[a] = global_sq_sum.get(a, {})
                specie_sum[sid][a] = specie_sum.get(a, {})
                specie_sq_sum[sid][a] = specie_sq_sum.get(a, {})
                for w, v in weights.items():
                    # global
                    global_sum[a][w] = global_sum[a].get(w, 0.0) + v
                    global_sq_sum[a][w] = global_sq_sum[a].get(w, 0.0) + v*v
                    # per-species
                    if not global_only:
                        specie_sum[sid][a][w] = specie_sum[sid][a].get(w, 0.0) + v
                        specie_sq_sum[sid][a][w] = specie_sq_sum[sid][a].get(w, 0.0) + v * v 
                
        ### compute stats    
        # global        
        for a in global_sum:
            self.brains_mean[a] = self.brains_mean.get(a, {})
            self.brains_std[a] = self.brains_std.get(a, {})
            for w in global_sum[a]:
                mean = global_sum[a][w] / global_count
                var = global_sq_sum[a][w] / global_count - mean * mean
                std = (var ** 0.5) + eps
                self.brains_mean[a][w] = gamma_mean * self.brains_mean[a].get(a, mean) + (1-gamma_mean) * mean
                self.brains_std[a][w] = gamma_std * self.brains_std[a].get(a, std) + (1-gamma_std) * std
        
        if global_only:
            return
        # per specie
        self.species_brains_mean = {}
        self.species_brains_std = {}
        for sid in specie_count:
            self.species_brains_mean[sid] = {}
            self.species_brains_std[sid] = {}
            cnt = specie_count[sid]
            for a in global_sum:
                self.species_brains_mean[sid][a] = self.species_brains_mean[sid].get(a, {})
                self.species_brains_std[sid][a] = self.species_brains_std[sid].get(a, {})
                for w in global_sum[a]:
                    if cnt == 0 or a not in specie_sum[sid]: # if empty specie
                        self.species_brains_mean[sid][a][w] = eps
                        self.species_brains_std[sid][a][w]  = eps
                        continue
                    m = specie_sum[sid][a][w] / cnt
                    v = specie_sq_sum[sid][a][w] / cnt - m * m
                    sd = (v ** 0.5) + eps
                    self.species_brains_mean[sid][a][w] = m
                    self.species_brains_std[sid][a][w]  = sd
                
    def dna_distance(self, dna1, dna2, eps=1e-9):
        v1, keys = self.dna_vector(dna1)
        v2, _    = self.dna_vector(dna2)
        stds = np.array([self.genes_std[k] for k in keys])
        weights = np.array([1.0 for k in keys]) # TODO weights for genes
        # normalized euclidean
        diff = (v1 - v2) / (stds + eps)
        euclid = np.sqrt(np.sum(weights * diff * diff))
        # cosine distance (directional)
        cosine = 1.0 - np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + eps)
        return euclid * (1.0 + self.alpha * cosine)    
        
    def choose_medoid(self, species: 'Species'):
        members = species.members
        best_dna = members[0].dna
        best_score = float("inf")
        for a in members:
            score = 0.0
            for b in members:
                if a is b:
                    continue
                score += self.dna_distance(a.dna, b.dna)
            if score < best_score:
                best_score = score
                best_dna = a.dna
        return best_dna
    
    def dna_vector(self, dna):
        keys = sorted(dna.gene_values.keys())
        return np.array([np.log1p(dna.gene_values[k]) for k in keys]), keys