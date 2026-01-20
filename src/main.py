from multiprocessing import shared_memory, Event, Process, Value, Queue
import numpy as np
import time
import yaml

from utils.arguments import *
from display.simulationWindow import SimulationWindow
from metrics.metrics import Metrics
from metrics.metricsWindow import MetricsWindow
from world.world import World
from display.frame import Frame
from optionHandler import ActionHandler
import utils.pos_utils as posUtils



if __name__=="__main__":
    ### Arguments
    args = parse_arguments()
    check_arguments(args)

    ### Configs
    config = yaml.load(open(args.config, 'r'), Loader=yaml.SafeLoader)
    world_config = yaml.load(open(config['path_config_world'], 'r'), Loader=yaml.SafeLoader)
    agent_config = yaml.load(open(config['path_config_agent'], 'r'), Loader=yaml.SafeLoader)
    display_config = yaml.load(open(config['path_config_display'], 'r'), Loader=yaml.SafeLoader)
    window_config = display_config['window']
    metrics_config = display_config['metrics']
    win_w, win_h = window_config["width"], window_config["height"]
    max_fps = window_config['max_fps']

    ### Shared Memory & events & queue
    shm = shared_memory.SharedMemory(create=True, size=win_w*win_h*3)
    shared_buf_name = shm.name
    cam_x = Value('i', 0)
    cam_y = Value('i', 0)
    event_ready = Event()
    event_close = Event()
    option_queue = Queue()
    action_queue = Queue()
    
    ### Options
    actionHandler = ActionHandler(option_queue, action_queue)

    ### Display Simulation process
    sim_window = SimulationWindow(win_w, win_h,
                                  world_config['worldgen']["width"], world_config['worldgen']["height"],
                                  window_config['cam_speed'], cam_x, cam_y, shared_buf_name, event_ready, event_close, option_queue, action_queue)
    sim_window_proc = Process(
        target = sim_window.run
    )
    sim_window_proc.start()

    ### Display Metrics process
    genes = list(agent_config['dna']['genes'].keys())
    actions = list(agent_config['brain']['utility_scores'].keys())
    metrics = Metrics(genes, step_freq=metrics_config['update_step_freq'])
    metrics_window = MetricsWindow(genes, actions, metrics.queue, event_close, time_freq= metrics_config['window_update_time_freq'])
    metrics_window_proc = Process(
        target = metrics_window.run
    )
    metrics_window_proc.start()


    ### Simulation
    posUtils.init(world_config['worldgen']['width'], world_config['worldgen']['height'], world_config['worldgen']['grid_cell_size'])
    world = World.from_config(world_config, agent_config)

    ### Frame
    frame = Frame(win_w, win_h, cam_x, cam_y, shm)

    ### Main loop
    prev_time = time.time_ns()
    nspf = 1/ max_fps * 1e9
    while not event_close.is_set():
        world.step()
        metrics.update(world)
        if time.time_ns() - prev_time > nspf:
            frame.render(world)
            actionHandler.update_actions(world)
            event_ready.set()
            actionHandler.update_options(frame)
            prev_time = time.time_ns()
            #time.sleep(0.001)s

    if sim_window_proc.is_alive():
        sim_window_proc.terminate()
    
    if metrics_window_proc.is_alive():
        metrics_window_proc.terminate()

