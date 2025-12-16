

class SimulationWindow():
    def __init__(self, window_w, window_h, world_w, world_h, cam_x, cam_y, shared_buf_name, event_ready, event_close, option_queue, action_queue) -> None:
        self.shared_buf_name = shared_buf_name
        self.window_w = window_w
        self.window_h = window_h
        self.world_w = world_w
        self.world_h = world_h
        self.event_ready = event_ready
        self.event_close = event_close
        self.option_queue = option_queue
        self.action_queue = action_queue
        self.cam_x = cam_x
        self.cam_y = cam_y
        self.cam_speed = 10
        self.zoom = 1.

    def run(self):
        import pygame, numpy as np
        from multiprocessing import shared_memory

        shm = shared_memory.SharedMemory(name=self.shared_buf_name)
        frame = np.ndarray((self.window_w, self.window_h, 3), dtype=np.uint8, buffer=shm.buf)

        pygame.init()
        screen = pygame.display.set_mode((self.window_w, self.window_h))

        camera_x = self.world_w//2 - self.window_w//2
        camera_y = self.world_h//2 - self.window_h//2
        while True:
            for e in pygame.event.get():
                if e.type == pygame.QUIT:
                    self.event_close.set()
                    return
                # actions
                if e.type == pygame.MOUSEBUTTONDOWN:
                    if e.button == 1:
                        x, y = e.pos
                        self.action_queue.put(("spawn_agent", (x+ self.cam_x.value, y+self.cam_y.value)))
                # options
                if e.type == pygame.KEYDOWN:
                    if e.key == pygame.K_r:
                        self.option_queue.put(('toggle_render', None))
                    if e.key == pygame.K_b:
                        self.option_queue.put(('toggle_biomes', None))
                    if e.key == pygame.K_a:
                        self.option_queue.put(('toggle_agents', None))
                    if e.key == pygame.K_f:
                        self.option_queue.put(('toggle_food', None))
                
            
            self.event_ready.wait()
            self.event_ready.clear()
            
            keys = pygame.key.get_pressed()
            
            # movement
            if keys[pygame.K_LEFT]:
                camera_x -= self.cam_speed
            if keys[pygame.K_RIGHT]:
                camera_x += self.cam_speed
            if keys[pygame.K_UP]:
                camera_y -= self.cam_speed
            if keys[pygame.K_DOWN]:
                camera_y += self.cam_speed
                
            # options
            
                
            self.cam_x.value = max(0, min(self.world_w - self.window_w, camera_x))
            self.cam_y.value = max(0, min(self.world_h - self.window_h, camera_y))
            
            pygame.surfarray.blit_array(screen, frame)
            pygame.display.flip()
        









