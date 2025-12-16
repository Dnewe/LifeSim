
import utils.pos_utils as posUtils
import numpy as np
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from world.world import World
    from agent.agent import Agent
    

# Idle Times after doing actions
WANDERING_IDLE_TIME = 10
EATING_IDLE_TIME = 0 #20
MATING_IDLE_TIME = 50
ATTACKING_IDLE_TIME = 10


class Goal():
    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs 
        self.completed = False

    def exec(self, agent: 'Agent', world:'World'):
        self.cost = 0
    
    
class IdleGoal(Goal):  
    def exec(self, agent: 'Agent', world:'World'):
        SENSE_COST_FACTOR = 0.1
        self.cost = agent.dna.idle_cost + agent.dna.sense_cost * SENSE_COST_FACTOR


class WanderGoal(Goal):
    def exec(self, agent: 'Agent', world:'World'):
        WANDER_SPEED_FACTOR = 0.5
        if agent.wander_pos is None:
            agent.wander_pos = posUtils.random_pos(agent.get_pos(), int(agent.dna.vision_range))
        reached = agent.walk(agent.wander_pos, WANDER_SPEED_FACTOR)
        if reached:
            agent.wander_pos = None
            agent.idle_time = WANDERING_IDLE_TIME
            self.completed = True
        SENSE_COST_FACTOR = 0.5
        self.cost = agent.dna.step_cost * WANDER_SPEED_FACTOR + agent.dna.sense_cost * SENSE_COST_FACTOR
        

class FollowGoal(Goal):
    def exec(self, agent: 'Agent', world:'World'):
        agent.walk(self.kwargs['partner'].get_pos())
        SENSE_COST_FACTOR = 0.5
        self.cost = agent.dna.step_cost + agent.dna.sense_cost * SENSE_COST_FACTOR


class EatGoal(Goal):
    def exec(self, agent: 'Agent', world:'World'):
        if self.kwargs['food_dist'] < agent.dna.size + world.foodmap.get_food_size(self.kwargs['food_pos']):
            self._eat(self.kwargs['food_pos'], agent, world)
            agent.idle_time = EATING_IDLE_TIME
            IDLE_COST_FACTOR = 1
            self.cost = agent.dna.idle_cost * IDLE_COST_FACTOR
            if world.foodmap.get_food_energy(self.kwargs['food_pos']) <=0 or agent.energy >= agent.dna.max_energy:
                self.completed = True
        else:
            agent.walk(self.kwargs['food_pos'])
            SENSE_COST_FACTOR = 1.
            self.cost = agent.dna.step_cost + agent.dna.sense_cost * SENSE_COST_FACTOR

    def _eat(self, food_pos, agent: 'Agent', world:'World'):
        energy = agent.dna.metabolism_efficiency * world.take_food_energy(food_pos)
        agent.satiety += agent.dna.satiety_gain * energy
        agent.energy += energy
        
    
class MateGoal(Goal):
    def exec(self, agent: 'Agent', world: 'World'):
        if self.kwargs['partner_dist'] < agent.dna.size + self.kwargs['partner'].dna.size:
            self._mate(agent, self.kwargs['partner'], world)
            SENSE_COST_FACTOR = 10.
            self.cost = agent.dna.sense_cost * SENSE_COST_FACTOR    # mating cost handled by agent.mate()
            self.completed = True
        else:
            agent.walk(self.kwargs['partner'].get_pos())
            SENSE_COST_FACTOR = 2.
            self.cost = agent.dna.step_cost + agent.dna.sense_cost       

    def _mate(self, agent1: 'Agent', agent2: 'Agent', world: 'World'):
        agent1.mate()
        agent2.mate()
        from agent.agent import Agent
        agent = Agent.from_parents(agent1, agent2)
        world.add_agent(agent)


class FleeGoal(Goal):
    def exec(self, agent: 'Agent', world: 'World'):
        pos_to_flee = self.kwargs['pos_to_flee']
        target_pos = posUtils.opposite(agent.get_pos(), pos_to_flee)
        agent.walk(target_pos)
        SENSE_COST_FACTOR = 2.
        self.cost = agent.dna.step_cost + agent.dna.sense_cost * SENSE_COST_FACTOR
    

class AttackGoal(Goal):
    """
    Attack based on dna distance
    """
    def exec(self, agent: 'Agent', world: 'World'):
        if self.kwargs['target_dist'] < agent.dna.size + self.kwargs['target'].dna.size:
            self._attack(agent, self.kwargs['target'], world)
            agent.idle_time = ATTACKING_IDLE_TIME
            STEP_COST_FACTOR = 3.
            SENSE_COST_FACTOR = 2.
            self.cost = agent.dna.step_cost * STEP_COST_FACTOR + agent.dna.sense_cost * SENSE_COST_FACTOR
            self.completed = True
        else:
            agent.walk(self.kwargs['target'].get_pos())
            SENSE_COST_FACTOR = 1.
            self.cost = agent.dna.step_cost + agent.dna.sense_cost * SENSE_COST_FACTOR
            
    def _attack(self, agent: 'Agent', target: 'Agent', world: 'World'):
        target.health -= agent.dna.damage