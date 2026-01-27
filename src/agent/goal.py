
import utils.pos_utils as posUtils
import numpy as np
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Dict, List, Type, Tuple, Any
if TYPE_CHECKING:
    from agent.agent import Agent
    

# Idle Times after doing actions
WANDERING_IDLE_TIME = 10
EATING_IDLE_TIME = 0 #20
REPRODUCE_IDLE_TIME = 100
ATTACKING_IDLE_TIME = 10


class Goal(ABC):
    cost: float
    args_keys: List[str] = []
    
    def __init__(self, args = {}) -> None:
        self.args = args
        self.completed = False

    @classmethod
    def valid_args(cls, args: Dict) -> bool:
        return True
    
    @abstractmethod
    def exec(self, agent: 'Agent') -> Tuple[str, Any] | None:
        ...
    
    
class IdleGoal(Goal):  
    def exec(self, agent: 'Agent'):
        SENSE_COST_FACTOR = 0.5
        self.cost = agent.genome.idle_cost + agent.genome.sense_cost * SENSE_COST_FACTOR


class WanderGoal(Goal):
    def exec(self, agent: 'Agent'):
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
    args_keys: List[str] = ['target']
    
    def exec(self, agent: 'Agent'):
        agent.walk(self.args['target'].get_pos())
        SENSE_COST_FACTOR = 0.5
        self.cost = agent.genome.step_cost + agent.genome.sense_cost * SENSE_COST_FACTOR
        
    @classmethod         
    def valid_args(cls, args) -> bool:
        return 'target' in args and args['target'] is not None


class EatGoal(Goal):
    args_keys: List[str] = ['food_dist', 'food_size', 'food_pos', 'food_energy']
    
    def exec(self, agent: 'Agent'):
        # eat food
        if self.args['food_dist'] < (agent.genome.size + self.args['food_size']):
            agent.idle_time = EATING_IDLE_TIME
            IDLE_COST_FACTOR = 1
            self.cost = agent.genome.idle_cost * IDLE_COST_FACTOR
            self.completed = True
            return self._eat(self.args['food_pos'], agent)
        # walk towards food
        else:
            agent.walk(self.args['food_pos'])
            SENSE_COST_FACTOR = 1.
            self.cost = agent.genome.step_cost + agent.genome.sense_cost * SENSE_COST_FACTOR

    def _eat(self, food_pos, agent: 'Agent'):
        amount = min(self.args['food_energy'], 10)
        energy = agent.genome.metabolism_efficiency * amount
        agent.energy += energy
        return ('eat', {'pos': food_pos, 'amount': amount})
    
    @classmethod         
    def valid_args(cls, args) -> bool:
        return 'food_pos' in args and args['food_pos'] is not None

        
class ReproduceGoal(Goal):
    args_keys: List[str] = ['can_reproduce']
    
    def exec(self, agent: 'Agent'):
        res = None
        if agent.genome.reproduction == 'clone':
            SENSE_COST_FACTOR = 10.
            self.cost = agent.genome.sense_cost * SENSE_COST_FACTOR    # cloning cost handled by agent.reproduce()
            if agent.can_reproduce:
                baby = self._clone(agent)
                res = ('add_agent', {'agent': baby})
                self.completed = True
        elif agent.genome.reproduction == 'mate':
            partner, partner_dist = self.args['reproduce_partner'], self.args['reproduce_partner_dist']
            if partner_dist < (agent.genome.size + partner.genome.size) / agent.genome.vision_range:
                if agent.can_reproduce and partner.can_reproduce:
                    baby = self._mate(agent, partner)
                    res = ('add_agent', {'agent': baby})
                    self.completed = True
                SENSE_COST_FACTOR = 10.
                self.cost = agent.genome.sense_cost * SENSE_COST_FACTOR    # mating cost handled by agent.reproduce()
                
            else:
                agent.walk(partner.get_pos())
                SENSE_COST_FACTOR = 1.
                self.cost = agent.genome.step_cost + agent.genome.sense_cost 
                return None
        return res
    
    @classmethod         
    def valid_args(cls, args) -> bool:
        return 'can_reproduce' in args and args['can_reproduce']
    
    def _clone(self, agent: 'Agent',):
        agent.reproduce()
        from agent.agent import Agent
        baby = Agent.clone(agent)
        return baby
        
    def _mate(self, agent1: 'Agent', agent2: 'Agent'):
        agent1.reproduce()
        agent2.reproduce()
        from agent.agent import Agent
        baby = Agent.from_parents(agent1, agent2)
        return baby
    

class FleeGoal(Goal):
    args_keys: List[str] = ['target']
    
    def exec(self, agent: 'Agent'):
        pos_to_flee = self.args['target'].get_pos()
        target_pos = posUtils.opposite(agent.get_pos(), pos_to_flee)
        agent.walk(target_pos)
        SENSE_COST_FACTOR = 2.
        self.cost = agent.genome.step_cost + agent.genome.sense_cost * SENSE_COST_FACTOR
        
    @classmethod         
    def valid_args(cls, args) -> bool:
        return 'target' in args and args['target'] is not None
    

class AttackGoal(Goal):
    """
    Attack based on genome distance
    """
    args_keys: List[str] = ['target', 'target_dist']
    
    def exec(self, agent: 'Agent'):
        if self.args['target_dist'] < (agent.genome.size + self.args['target'].genome.size):
            self._attack(agent, self.args['target'])
            agent.idle_time = ATTACKING_IDLE_TIME
            STEP_COST_FACTOR = 3.
            SENSE_COST_FACTOR = 2.
            self.cost = agent.genome.step_cost * STEP_COST_FACTOR + agent.genome.sense_cost * SENSE_COST_FACTOR
            self.completed = True
        else:
            agent.walk(self.args['target'].get_pos())
            SENSE_COST_FACTOR = 1.
            self.cost = agent.genome.step_cost + agent.genome.sense_cost * SENSE_COST_FACTOR
            
    def _attack(self, agent: 'Agent', target: 'Agent'):
        target.health -= agent.genome.damage
        
    @classmethod         
    def valid_args(cls, args) -> bool:
        return 'target' in args and args['target'] is not None
        

class PheromoneGoal(Goal):
    pass
        
        
GOALS_MAP: Dict[str, Type[Goal]] = {
    'idle': IdleGoal,
    'wander': WanderGoal,
    'follow': FollowGoal,
    'eat': EatGoal,
    'reproduce': ReproduceGoal,
    'flee': FleeGoal,
    'attack': AttackGoal,
    'pheromone': PheromoneGoal
}


def valid_action_args(action:str, args: Dict[str, Any]):
    return GOALS_MAP[action].valid_args(args)
