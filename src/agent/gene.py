from typing import Literal, Dict, Self
import yaml
import numpy as np


class Attribute():
    def __init__(self, mod:Literal['log2','pow2']|None = None,
                min:float|None =None, max:float|None =None, factor:float =1, offset:float =0) -> None:
        self.min_v = min
        self.max_v = max
        self.factor = factor
        self.offset = offset
        self.value = 0
        self.default = 0
        self._set_modifier(mod)
        
    def _set_modifier(self, mod: str|None):
        if mod == 'log2':
            self._fn = np.log2
        if mod == 'pow2':
            self._fn = lambda x: x*x
        else:
            self._fn = lambda x: x

    def compute_value(self, value: float) -> float:
        value *= self.factor
        value += self.offset
        value = self._fn(value)
        if self.min_v is not None:
            value = max(self.min_v, value)
        if self.max_v is not None:
            value = min(self.max_v, value) 
        return value
    
    def set_default(self, value: float) -> None:
        self.default = self.compute_value(value)

    def set_value(self, value: float) -> None:
        self.value = self.compute_value(value)




class Gene():
    def __init__(self, attributes:Dict[str,Attribute], value:float, default:float = 1., mutable:bool = True, mutation_scale_factor:float = 1.) -> None:
        self.attributes = {name: attr for name, attr in attributes.items()}
        self.mutable = mutable
        self.mutation_scale_factor = mutation_scale_factor # TODO use
        self.default = default
        self.value = self.value
        self._init_attr_values()
        
    def _init_attr_values(self):
        for attr in self.attributes.values():
            attr.set_default(self.default)
            attr.set_value(self.value)

    def get_attr_value(self, name:str):
        return self.attributes[name].value
    
    def get_attr_default(self, name:str):
        return self.attributes[name].default


class DiscreteGene(Gene):
    pass


class ContinuousGene(Gene):
    def __init__(self, attributes:Dict[str,Attribute], value:float, default:float = 1., mutable:bool = True, mutation_scale_factor:float = 1.) -> None:
        self.attributes = {name: attr for name, attr in attributes.items()}
        self.mutable = mutable
        self.mutation_scale_factor = mutation_scale_factor # TODO use
        self.default = default
        self.value = self.value
        self._init_attr_values()
        
    def _init_attr_values(self):
        for attr in self.attributes.values():
            attr.set_default(self.default)
            attr.set_value(self.value)

    def get_attr_value(self, name:str):
        return self.attributes[name].value
    
    def get_attr_default(self, name:str):
        return self.attributes[name].default



class DNA():
    mode: Literal['mean','rdm_choice']
    genes: Dict[str, Gene]
    
    def from_config(self, dna_config):
        self.mode = dna_config['mode']
        self.genes = {}
        for gene_k, gene_v in dna_config['genes'].items():
            attributes = {}
            for attr_k, attr_v in gene_v['attributes']:
                attributes[attr_k] = Attribute(mod= attr_v['mod'], 
                                                  min= attr_v['min'], 
                                                  max= attr_v['max'], 
                                                  factor= attr_v['factor'],
                                                  offset= attr_v['offset'])
            self.genes[gene_k] = Gene(attributes= attributes, 
                                      value= gene_v['default'], 
                                      default= gene_v['default'],
                                      mutable= gene_v['mutable'], 
                                      mutation_scale_factor = gene_v['mutation_scale_factor'])
            
    def _compute_new_gene_value(self):
        pass
            
    def from_dna1_dna2(self, dna1: Self, dna2: Self):
        # get mode from either parents
        self.mode = np.random.choice([dna1.mode, dna2.mode])
        # extract genes from parents
        self.genes = {}
        for gene_k in dna1.genes.keys():
            gene1, gene2 = dna1.genes[gene_k], dna2.genes[gene_k]
            gene_value = (gene1.value + gene2.value) / 2
            gene = Gene(gene1.attributes, gene_value, gene1.default, gene1.mutable, gene1.mutation_scale_factor)
            self.genes[gene_k] = gene



morphology_attributes = {
    'size': Attribute(mod='log2', min=1, factor=5), # default = 5
    'max_energy': Attribute(min=1, factor=1000) # default = 
}

morphology_gene = Gene(morphology_attributes, 1, True)



if __name__ == '__main__':
    gene_values = [1, 0.5, 1.5]
    
    for v in gene_values:
        morphology_gene.value = v
        for attr_name in morphology_gene.attributes.keys():
            print()