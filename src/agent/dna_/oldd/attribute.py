from typing import List


class Attribute():
    # value of the attribute computed from the gene value
    value: float|str|bool
    
    def from_config(self, config):
        pass
    
    def copy(self, attr):
        pass


class ContinuousAttribute(Attribute):
    min_value: float
    max_value: float
    factor: float
    offset: float


class DiscreteAttribute(Attribute):
    value_weights: List[float]