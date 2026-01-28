from display.frame import Frame
from world.world import World
from sprite.agent import Agent


class ActionHandler():
    # display options
    
    def __init__(self, option_queue, action_queue) -> None:
        self.option_queue = option_queue
        self.action_queue = action_queue
        
    def update_options(self, frame: Frame, world: World):
        while not self.option_queue.empty():
            cmd, value = self.option_queue.get()
            b = None
            if cmd == 'toggle_render':
                b = not frame.show_biomes
                frame.show_agents = b
                frame.show_food = b
                frame.show_biomes = b
            if cmd == 'toggle_biomes':
                b = not frame.show_biomes
                frame.show_biomes = b
            elif cmd == 'toggle_agents':
                b = not frame.show_agents
                frame.show_agents = b
            elif cmd == 'toggle_food':
                b = not frame.show_food
                frame.show_food = b
            elif cmd == 'toggle_world_pause':
                b = not world.paused
                world.paused = b
            elif cmd == 'toggle_speciation_pause':
                b =  not world.speciator.paused
                world.speciator.paused = b
            print(f'{cmd}: {b}')
                
    
    def update_actions(self, world: World):
        while not self.action_queue.empty():
            cmd, value = self.action_queue.get()
            
            if cmd == 'spawn_agent':
                x, y = value
                agent = Agent.from_config(world.agent_config)
                agent.x = x
                agent.y = y
                world.add_agent(agent)
        
    