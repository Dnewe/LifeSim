
import utils.pos_utils as posUtils
import numpy as np
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Dict, List, Type
if TYPE_CHECKING:
    from world.world import World
    from agent.agent import Agent
    

# Idle Times after doing actions
WANDERING_IDLE_TIME = 10
EATING_IDLE_TIME = 0 #20
REPRODUCE_IDLE_TIME = 50
ATTACKING_IDLE_TIME = 10


class Goal(ABC):
    var_names = []
    cost: float
    
    def __init__(self, context={}) -> None:
        self.context = context 
        self.completed = False

    @classmethod
    def is_valid(cls, context) -> bool:
        return all(context[v] is not None for v in cls.var_names)
    
    @abstractmethod
    def exec(self, agent: 'Agent', world:'World'):
        ...
    
    
class IdleGoal(Goal):  
    def exec(self, agent: 'Agent', world:'World'):
        SENSE_COST_FACTOR = 0.5
        self.cost = agent.genome.idle_cost + agent.genome.sense_cost * SENSE_COST_FACTOR


class WanderGoal(Goal):
    def exec(self, agent: 'Agent', world:'World'):
        WANDER_SPEED_FACTOR = 0.5
        if agent.wander_pos is None:
            agent.wander_pos = posUtils.random_pos(agent.get_pos(), int(agent.genome.vision_range))
        reached = agent.walk(agent.wander_pos, WANDER_SPEED_FACTOR)
        if reached:
            agent.wander_pos = None
            agent.idle_time = WANDERING_IDLE_TIME
            self.completed = True
        SENSE_COST_FACTOR = 0.5
        self.cost = agent.genome.step_cost * WANDER_SPEED_FACTOR + agent.genome.sense_cost * SENSE_COST_FACTOR
        

class FollowGoal(Goal):
    var_names = ['nearest_ally']
    def exec(self, agent: 'Agent', world:'World'):
        agent.walk(self.context[self.var_names[0]].get_pos())
        SENSE_COST_FACTOR = 0.5
        self.cost = agent.genome.step_cost + agent.genome.sense_cost * SENSE_COST_FACTOR


class EatGoal(Goal):
    var_names = ['nearest_food', 'nearest_food_dist']
    def exec(self, agent: 'Agent', world:'World'):
        if self.context[self.var_names[1]] + 0.5 < (agent.genome.size + world.foodmap.get_food_size(self.context[self.var_names[0]])) / agent.genome.vision_range:
            self._eat(self.context[self.var_names[0]], agent, world)
            agent.idle_time = EATING_IDLE_TIME
            IDLE_COST_FACTOR = 1
            self.cost = agent.genome.idle_cost * IDLE_COST_FACTOR
            if world.foodmap.get_food_energy(self.context[self.var_names[0]]) <=0 or agent.energy >= agent.genome.max_energy:
                self.completed = True
        else:
            agent.walk(self.context[self.var_names[0]])
            SENSE_COST_FACTOR = 1.
            self.cost = agent.genome.step_cost + agent.genome.sense_cost * SENSE_COST_FACTOR

    def _eat(self, food_pos, agent: 'Agent', world:'World'):
        energy = agent.genome.metabolism_efficiency * world.take_food_energy(food_pos)
        agent.satiety += agent.genome.satiety_gain * energy
        agent.energy += energy


        
class ReproduceGoal(Goal):
    #var_names = ['nearest_ally', 'nearest_ally_dist'] TODO 
    def exec(self, agent: 'Agent', world: 'World'):
        if agent.genome.reproduction == 'clone':
            if agent.ready_to_reproduce:
                self._clone(agent, world)
                self.completed = True
            SENSE_COST_FACTOR = 10.
            self.cost = agent.genome.sense_cost * SENSE_COST_FACTOR    # cloning cost handled by agent.reproduce()
        elif agent.genome.reproduction == 'mate':
            partner, partner_dist = self.context[self.var_names[0]], self.context[self.var_names[1]] + 0.5 # TODO
            if partner_dist < (agent.genome.size + partner.genome.size) / agent.genome.vision_range:
                if agent.ready_to_reproduce and partner.ready_to_reproduce:
                    self._mate(agent, partner, world)
                    self.completed = True
                SENSE_COST_FACTOR = 10.
                self.cost = agent.genome.sense_cost * SENSE_COST_FACTOR    # mating cost handled by agent.reproduce()
            else:
                agent.walk(partner.get_pos())
                SENSE_COST_FACTOR = 1.
                self.cost = agent.genome.step_cost + agent.genome.sense_cost 
    
    @classmethod         
    def is_valid(cls, context) -> bool:
        return context['ready_to_reproduce']
    
    def _clone(self, other: 'Agent', world: 'World'):
        other.reproduce()
        from agent.agent import Agent
        agent = Agent.clone(other)
        world.add_agent(agent)
        
    def _mate(self, agent1: 'Agent', agent2: 'Agent', world: 'World'):
        agent1.reproduce()
        agent2.reproduce()
        from agent.agent import Agent
        agent = Agent.from_parents(agent1, agent2)
        world.add_agent(agent)
    

class FleeGoal(Goal):
    var_names = ['nearest_enemy']
    def exec(self, agent: 'Agent', world: 'World'):
        pos_to_flee = self.context[self.var_names[0]].get_pos()
        target_pos = posUtils.opposite(agent.get_pos(), pos_to_flee)
        agent.walk(target_pos)
        SENSE_COST_FACTOR = 2.
        self.cost = agent.genome.step_cost + agent.genome.sense_cost * SENSE_COST_FACTOR
    

class AttackGoal(Goal):
    """
    Attack based on genome distance
    """
    var_names = ['nearest_enemy', 'nearest_enemy_dist']
    def exec(self, agent: 'Agent', world: 'World'):
        if self.context[self.var_names[1]] + 0.5 < (agent.genome.size + self.context[self.var_names[0]].genome.size)/ agent.genome.vision_range:
            self._attack(agent, self.context[self.var_names[0]])
            agent.idle_time = ATTACKING_IDLE_TIME
            STEP_COST_FACTOR = 3.
            SENSE_COST_FACTOR = 2.
            self.cost = agent.genome.step_cost * STEP_COST_FACTOR + agent.genome.sense_cost * SENSE_COST_FACTOR
            self.completed = True
        else:
            agent.walk(self.context[self.var_names[0]].get_pos())
            SENSE_COST_FACTOR = 1.
            self.cost = agent.genome.step_cost + agent.genome.sense_cost * SENSE_COST_FACTOR
            
    def _attack(self, agent: 'Agent', target: 'Agent'):
        target.health -= agent.genome.damage
        
        
GOALS_MAP: Dict[str, Type[Goal]] = {
    'idle': IdleGoal,
    'wander': WanderGoal,
    'follow': FollowGoal,
    'eat': EatGoal,
    'reproduce': ReproduceGoal,
    'flee': FleeGoal,
    'attack': AttackGoal,
}


def valid_action(action:str, context):
    return GOALS_MAP[action].is_valid(context)
