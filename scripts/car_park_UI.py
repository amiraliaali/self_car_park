import numpy as np
from car_park import *
import cv2 as cv
import pygame
import environment as env
import torch
import sys
import time

class CarParkUI(CarPark):
    def __init__(self):
        super().__init__()

    def test_run(self, training_episodes):
        self.train_deep_sarsa(episodes=training_episodes)
        self.show_test_agent()

    def show_test_agent(self):
        env.reset()
        
        # Initialize OpenCV video writer
        # Set video properties like frame width, height, and frame rate
        frame_width, frame_height = env.WIDTH, env.HEIGHT  # Define your window size
        fourcc = cv.VideoWriter_fourcc(*'H264')  # Use H264 codec for MP4
        out = cv.VideoWriter('output_video.mp4', fourcc, env.FPS, (frame_width, frame_height))

        running = True
        start_time = time.time()

        while running:
            if time.time() - start_time > 10:  # Run for 10 seconds
                running = False

            state = torch.FloatTensor(env.get_current_state()) 
            av = self.q_network(state).detach()
            action = torch.argmax(av, dim=-1, keepdim=False).unsqueeze(0).item()

            # keys = env.ACTION_KEY_MAPPING[action]
            # pygame.event.post(pygame.event.Event(pygame.KEYDOWN, {'key': keys}))
            # pygame.event.post(pygame.event.Event(pygame.KEYUP, {'key': keys}))

            # for event in pygame.event.get():
            #     if event.type == pygame.QUIT:
            #         running = False

            env.move_car(action)

            if env.check_collision() or env.check_parking():
                running = False

            # Draw everything
            env.draw_environment()
            pygame.display.flip()

            # Capture the current screen
            frame = pygame.surfarray.array3d(pygame.display.get_surface())  # Convert surface to numpy array
            frame = np.transpose(frame, (1, 0, 2))  # Convert from (width, height, channels) to (height, width, channels)
            frame = cv.cvtColor(frame, cv.COLOR_RGB2BGR)  # Convert RGB to BGR for OpenCV

            # Write the frame to the video file
            out.write(frame)

            env.clock.tick(env.FPS)

        # Release the video writer and clean up
        out.release()
        pygame.quit()
        sys.exit()

if __name__ == "__main__":
    cp = CarParkUI()
    cp.test_run(100)
