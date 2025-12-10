from typing import Tuple, List
import numpy as np
import copy


SIZE_FACTOR = 0.2
MIN_SIZE = 0.1
VISION_RANGE_FACTOR = 0.01





class DNA():
    # physical
    r: float
    g: float
    b: float
    size: float
    speed: float

    # senses
    vision_range: float

    # energy
    max_energy: float
    energy_threshold: float

    # mating
    min_energy_to_mate:int
    min_age_to_mate:int
    random_scale:float


    def __init__(self) -> None:
        pass
    
    def from_config(self, config, random=True):
        """
        Set attributes from config.
        Normally randomized
        """
        # mutable
        #self.r, self.g, self.b = config['agent_color']
        self.size = config['agent_size']
        self.base_max_age = config['agent_max_age']
        self.base_speed = config['agent_speed']
        self.flee_distance = config['agent_flee_distance']
        self.base_vision_range = config['agent_vision_range']
        self.base_max_energy = config['agent_max_energy']
        self.energy_ratio_threshold = config['agent_energy_ratio_threshold']
        self.mutation_scale = config['agent_mutation_scale']
        # immutable
        self._base_dna = copy.deepcopy(self) # keep base config mutable dna
        self._min_energy_ratio_to_mate = config['agent_min_energy_ratio_to_mate']
        self._min_age_ratio_to_mate = config['agent_min_age_ratio_to_mate']
        self._mating_cooldown = config['agent_mating_cooldown']
        random_scale = config['attr_random_scale']
        if random: self.randomize_all_attr(random_scale)
        self.adjust_attributes()

    def from_dna1_dna2(self, dna1, dna2, random=True):
        """
        Set attributes from two parent DNAs.
        Resulting attributes are the mean of the two, 
        then normally randomized
        """
        self._base_dna = dna1._base_dna
        for k in self._get_attr_keys(dna1, skip='_'):
            v1 = getattr(dna1, k)
            v2 = getattr(dna2, k)
            setattr(self, k, (v1+v2)/2)
        for k in self._get_attr_keys(dna1, keep='_'):
            setattr(self, k, getattr(dna1, k))
        if random: self.randomize_all_attr(self.mutation_scale)
        self.adjust_attributes()

    def adjust_attributes(self):
        self.size = max(MIN_SIZE, self.size)
        self.speed = self.base_speed / np.sqrt(self.size * SIZE_FACTOR)
        self.max_energy = self.base_max_energy * (self.size * SIZE_FACTOR)
        self.vision_range = self.base_vision_range / (self.size * SIZE_FACTOR)
        self.max_age = self.base_max_age * (self.size * SIZE_FACTOR)
        self.min_age_to_mate = self.max_age * self._min_age_ratio_to_mate
        self.min_energy_to_mate = self.max_energy * self._min_energy_ratio_to_mate
        self.energy_threshold = self.max_energy * self.energy_ratio_threshold

    def randomize_all_attr(self, scale):
        for k in self._get_attr_keys(skip='_'):
            setattr(self, k, self._random(getattr(self, k), scale))

    def _random(self, value, scale, positive=True):
        if type(value) not in [int, float]:
            return value
        value = value * (1 + np.random.normal(0, scale))
        return max(0, value) if positive else value
    
    def _get_attr_keys(self, dna=None, skip=None, keep=None):
        dna = self if dna is None else dna
        return [k for k in vars(dna).keys() 
                if (skip is None or k[0:len(skip)]!=skip) 
                    and 
                    (keep is None or k[0:len(keep)]==keep)]
    
    def distance_from(self, dna):
        dists = []
        for k in self._get_attr_keys(dna, skip='_'):
            v1 = getattr(self, k)
            v2 = getattr(dna, k)
            dists.append(2*np.abs(v2-v1)/(v2+v1))
        return np.mean(dists)
    
    def distance_from_default(self):
        return self.distance_from(self._base_dna)