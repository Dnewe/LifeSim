from typing import Dict, Literal, Self
from agent.dna_.oldd.gene_ import Gene
import numpy as np


class DNA():
    # way to merge two parent dnas into a child dna
    merge_mode: Literal['mean', 'choice']
    # gene dict (gene_name, gene_obj)
    genes: Dict[str, Gene]
    # scale of the normal distrubition used during random mutation
    mutation_scale: float
    
    ### Attributes
    # physical
    size: float
    speed: float
    # physiological
    lifespan: float
    max_energy: float
    energy_threshold: float
    energy_to_mate: float
    age_to_mate: float
    # sensory
    vision_range: float
    # energy costs
    cost_factor: float
    step_cost: float
    sense_cost: float
    # constants
    mating_cooldown: float
    
    
    def __init__(self, genes, merge_mode, mutation_scale) -> None:
        self.genes = genes
        self.merge_mode = merge_mode
        self.mutation_scale = mutation_scale
    
    @classmethod
    def from_config(cls, config):
        genes = {k: Gene.from_config(v) for k,v in config['genes']}
        return cls(genes, config['merge_mode'], config['mutation_scale'])
    
    @classmethod
    def from_parents(cls, dna1:Self, dna2:Self):
        merge_mode = np.random.choice([dna1.merge_mode, dna2.merge_mode])
        mutation_scale = np.random.choice([dna1.mutation_scale, dna2.mutation_scale])
        genes = {k: Gene.from_parents(gene1, gene2, merge_mode) for (k, gene1), (_, gene2) in zip(dna1.genes.items(), dna2.genes.items())}
        return cls(genes, merge_mode, mutation_scale)
    
    def compute_attributes(self):
        for gene in self.genes.values():
            gene.compute_attributes()
        self._set_attributes()
    
    def _set_attributes(self):
        """
        set attributes values from genes to dna object, 
        if multiple genes changes the same attribute:
            if value int/float : sum them
            else : random choice
        """
        for gene_v in self.genes.values():
            for k, v in gene_v.attributes.items():
                if hasattr(self, k):
                    if type(v) == int or type(v) == float:
                        setattr(self, k, getattr(self, k) + v)
                        print(f'debug: sum attr {k} values')    # TODO remove if works properly
                    else:
                        setattr(self, k, np.random.choice([getattr(self, k), v]))
                        print(f'debug: random choice attr {k} values')  # TODO remove if works properly
                else:
                    setattr(self, k, v)
    
    def mutate(self, scale_factor=1.):
        # TODO other types of mutation
        for gene in self.genes.values():
            if gene.mutable:
                gene.random_mutation(self.mutation_scale * scale_factor)
                