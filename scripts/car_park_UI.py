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
        print("Starting training...")
        self.train_actor_critic(episodes=training_episodes)
        print("Training completed. Running test agent...")
        self.show_test_agent()

    def show_test_agent(self):
        self.environment_inst.env_reset()

        # Video writer setup
        frame_width, frame_height = env.WIDTH, env.HEIGHT
        fourcc = cv.VideoWriter_fourcc(*'H264')  # Codec for MP4
        out = cv.VideoWriter('output_video.mp4', fourcc, env.FPS, (frame_width, frame_height))

        running = True
        start_time = time.time()

        while running:
            if time.time() - start_time > 30:  # Run for 30 seconds
                running = False

            # Get current state and ensure it's valid
            state = self.environment_inst.get_current_state()
            state = torch.FloatTensor(state).unsqueeze(0)

            # Use policy network for action selection
            with torch.no_grad():
                action_probs, _ = self.model(state)
                action = torch.argmax(action_probs).item()

            # Execute the action
            self.environment_inst.move_car(action)

            # Check for terminal conditions
            if self.environment_inst.check_collision() or self.environment_inst.check_parking():
                running = False

            # Draw environment in pygame
            self.environment_inst.draw_environment()
            pygame.display.flip()

            # Capture the current screen for the video
            frame = pygame.surfarray.array3d(pygame.display.get_surface())
            frame = np.transpose(frame, (1, 0, 2))  # (width, height, channels) -> (height, width, channels)
            frame = cv.cvtColor(frame, cv.COLOR_RGB2BGR)  # Convert RGB to BGR for OpenCV
            out.write(frame)

            env.CLOCK.tick(env.FPS)

        # Clean up
        print("Simulation complete. Saving video.")
        out.release()
        pygame.quit()
        sys.exit()

if __name__ == "__main__":
    cp = CarParkUI()
    print("Starting training...")
    cp.train_actor_critic(episodes=5)
    print("Training completed. Running test agent...")
    # cp.load_model()
    # cp.save_model()
    cp.show_test_agent()
