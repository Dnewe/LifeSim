from multiprocessing import shared_memory, Event, Process, Value
import numpy as np
import time
import yaml

from utils.arguments import *
from display.simulationWindow import SimulationWindow
from metrics.metrics import Metrics
from metrics.metricsWindow import MetricsWindow
from world import World
from display.frame import Frame
import utils.pos_utils as posUtils


if __name__=="__main__":
    ### Arguments
    args = parse_arguments()
    check_arguments(args)

    ### Configs
    config = yaml.load(open(args.config, 'r'), Loader=yaml.SafeLoader)
    world_config = yaml.load(open(config['path_config_world'], 'r'), Loader=yaml.SafeLoader)
    agent_config = yaml.load(open(config['path_config_agent'], 'r'), Loader=yaml.SafeLoader)
    window_config = config['window']
    metrics_config = config['metrics']
    win_w, win_h = window_config["width"], window_config["height"]

    ### Shared Memory & events
    shm = shared_memory.SharedMemory(create=True, size=win_w*win_h*3)
    shared_buf_name = shm.name
    cam_x = Value('i', 0)
    cam_y = Value('i', 0)
    event_ready = Event()
    event_close = Event()

    ### Display Simulation process
    sim_window = SimulationWindow(win_w, win_h,
                                  world_config["width"], world_config["height"],
                                  cam_x, cam_y, shared_buf_name, event_ready, event_close)
    sim_window_proc = Process(
        target = sim_window.run
    )
    sim_window_proc.start()

    ### Display Metrics process
    genes = list(agent_config['dna']['genes'].keys())
    metrics = Metrics(genes, step_freq=metrics_config['update_step_freq'])
    metrics_window = MetricsWindow(genes, metrics.queue, event_close, time_freq= metrics_config['window_update_time_freq'])
    metrics_window_proc = Process(
        target = metrics_window.run
    )
    metrics_window_proc.start()


    ### Simulation
    posUtils.init(world_config['width'], world_config['height'], world_config['grid_cell_size'])
    world = World.from_config(world_config, agent_config)

    ### Frame
    frame = Frame(win_w, win_h, cam_x, cam_y, shm)

    ### Main loop
    while not event_close.is_set():
        world.step()
        frame.draw(world)
        metrics.update(world)
        event_ready.set()
        #time.sleep(0.001)

    if sim_window_proc.is_alive():
        sim_window_proc.terminate()
    
    if metrics_window_proc.is_alive():
        metrics_window_proc.terminate()

