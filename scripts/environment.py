import pygame
import math
import sys
import time
import random
import cv2 as cv
import numpy as np
from car import Car
import matplotlib.pyplot as plt

ACTIONS_MAPPING = {
    "0": "Accelerate straight",
    "1": "Accelerate left",
    "2": "Accelerate right",
    "3": "Decelerate",
    "4": "Brake left",
    "5": "Brake right",
    "6": "Do nothing",
}

REWARDS = {
    "collision": -1000,
    "parked": 1000,
    "time_up": -1000,
}

pygame.init()

WIDTH, HEIGHT = 800, 600

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY = (200, 200, 200)
RED = (255, 0, 0)
GREEN = (0, 255, 0)

CLOCK = pygame.time.Clock()
FPS = 60


class Environment:
    def __init__(self) -> None:
        self.generate_car()
        self.generate_obstacle_cars()
        self.iteration_num = 0
        self.parking_spot_x, self.parking_spot_y = 300, 300
        self.window_opened = False
        self.action_num = 0
        self.parked_tolerance_margin = 10
        self.env_reset()
        self.all_actions_current_run = list()
        self.gaussian_reward_plane = self.create_gaussian_plane()
        self.save_gaussian_graph()

    def generate_car(self):
        self.car_agent = Car(50, 400)
        self.car_surface = pygame.Surface(
            (self.car_agent.width, self.car_agent.height), pygame.SRCALPHA
        )
        self.car_surface.fill(BLACK)

    def create_gaussian_plane(self):
        # Generate coordinate grid
        x = np.arange(0, WIDTH, 1)
        y = np.arange(0, HEIGHT, 1)
        x, y = np.meshgrid(x, y)

        # Define Gaussian parameters
        mu_x = self.parking_spot_x
        mu_y = self.parking_spot_y
        sigma_x = WIDTH / 5
        sigma_y = HEIGHT / 5

        # Compute Gaussian distribution
        gaussian = 2 * np.exp(
            -(((x - mu_x) ** 2) / (2 * sigma_x ** 2) + ((y - mu_y) ** 2) / (2 * sigma_y ** 2))
        )
        return gaussian
    
    def save_gaussian_graph(self, filename="gaussian_reward_graph.png"):
        plt.figure(figsize=(8, 6))
        plt.imshow(self.gaussian_reward_plane, extent=[0, WIDTH, 0, HEIGHT], origin='lower', cmap='viridis')
        plt.colorbar(label="Intensity")
        plt.title("Gaussian Distribution")
        plt.xlabel("X-axis")
        plt.ylabel("Y-axis")
        plt.savefig(filename)
        plt.close()

    def generate_obstacle_cars(self):
        self.obstacle_cars = [
            # pygame.Rect(400, 400, CAR_WIDTH, CAR_HEIGHT),
            # pygame.Rect(400, 350, CAR_WIDTH, CAR_HEIGHT),
            # pygame.Rect(400, 250, CAR_WIDTH, CAR_HEIGHT),
            # pygame.Rect(400, 200, CAR_WIDTH, CAR_HEIGHT),
        ]

    def env_reset(self, reset_actions_list=True):
        # self.car_x, self.car_y = random.randint(50, 100), random.randint(50, 500)
        pygame.init()
        self.car_agent.reset()
        self.current_car_parking_distance = math.inf
        self.total_moves = 0
        pygame.event.clear()
        if reset_actions_list:
            self.all_actions_current_run = []
            self.iteration_num += 1
        return self.get_current_state()

    def generate_window(self):
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("Parallel Parking Simulator")

    def draw_environment(self):
        if not self.window_opened:
            self.generate_window()
            self.window_opened = True

        self.screen.fill(GRAY)
        pygame.draw.rect(
            self.screen,
            GREEN,
            (
                self.parking_spot_x,
                self.parking_spot_y,
                self.car_agent.width,
                self.car_agent.height,
            ),
            2,
        )
        for obstacle in self.obstacle_cars:
            pygame.draw.rect(self.screen, RED, obstacle)

        rotated_car = pygame.transform.rotate(self.car_surface, -self.car_agent.angle)
        car_rect = rotated_car.get_rect(center=(self.car_agent.x, self.car_agent.y))
        self.screen.blit(rotated_car, car_rect.topleft)

    def move_car(self, action):
        if action == 0:  # Accelerate
            self.car_agent.increase_speed()

        elif action == 1:  # Decelerate
            self.car_agent.decrease_speed()

        elif action == 2:  # Left
            self.car_agent.steer_left()

        else:  # Right
            self.car_agent.steer_right()

        self.car_agent.update()

    def check_collision(self):
        """Check for collisions with obstacles or boundaries."""
        car_rect = pygame.Rect(
            self.car_agent.x - self.car_agent.width // 2,
            self.car_agent.y - self.car_agent.height // 2,
            self.car_agent.width,
            self.car_agent.height,
        )
        if self.car_agent.boundary_collision(WIDTH, HEIGHT):
            return True
        for obstacle in self.obstacle_cars:
            if car_rect.colliderect(obstacle):
                return True
        return False

    def check_parking(self):
        """Check if the car is parked in the designated spot and centered inside."""
        return self.car_agent.check_parking(
            self.parking_spot_x, self.parking_spot_y, self.parked_tolerance_margin
        )

    def check_getting_closer(self):
        """Check if the car is getting closer to the parking spot."""
        distance_to_parking = self.car_agent.distance_to_parking(
            self.parking_spot_x, self.parking_spot_y
        )

        if distance_to_parking < self.current_car_parking_distance:
            self.current_car_parking_distance = distance_to_parking
            return True
        return False

    def calculate_distance_reward(self):
        """Calculates the reward based on the position in the gaussian plane."""
        x, y = self.car_agent.x, self.car_agent.y
        reward = self.gaussian_reward_plane[int(y), int(x)]
        return reward

    def generate_video_current_run(self):
        self.env_reset(reset_actions_list=False)
        frame_width, frame_height = WIDTH, HEIGHT
        fourcc = cv.VideoWriter_fourcc(*"H264")  # Codec for MP4
        out = cv.VideoWriter(
            f"output_video_{self.iteration_num}.mp4",
            fourcc,
            FPS,
            (frame_width, frame_height),
        )

        for i in self.all_actions_current_run:
            state = self.get_current_state()
            action = i
            self.move_car(action)

            # Draw environment in pygame
            self.draw_environment()
            pygame.display.flip()

            # Capture the current screen for the video
            frame = pygame.surfarray.array3d(pygame.display.get_surface())
            frame = np.transpose(
                frame, (1, 0, 2)
            )  # (width, height, channels) -> (height, width, channels)
            frame = cv.cvtColor(
                frame, cv.COLOR_RGB2BGR
            )  # Convert RGB to BGR for OpenCV
            out.write(frame)

            CLOCK.tick(FPS)
        print("Finished saving video!!")
        out.release()
        self.all_actions_current_run = []
        self.screen.fill(GRAY)

    def execute_move(self, action):
        """Execute the given action, return (done, reward, next_state)."""
        self.all_actions_current_run.append(action)
        # Update the car's state based on the action
        self.move_car(action)

        # Get the updated state
        state = self.get_current_state()

        self.total_moves += 1

        # Collision detection
        if self.check_collision():
            return True, REWARDS["collision"], state

        # Parking success
        if self.check_parking():
            print("parked succesfully")
            self.generate_video_current_run()
            return True, REWARDS["parked"], state

        if self.total_moves > 1000:
            self.total_moves = 0
            return True, REWARDS["time_up"], state

        reward = self.calculate_distance_reward()

        # Default penalty for no progress
        return False, reward, state

    def get_current_state(self):
        return [
            self.car_agent.x,
            self.car_agent.y,
            self.car_agent.angle,
            self.parking_spot_x,
            self.parking_spot_y,
        ]

    def test_run(self, run_time_in_sec):
        self.env_reset()

        running = True
        start_time = time.time()

        while running:
            if time.time() - start_time > run_time_in_sec:
                running = False

            random_action = random.randint(0, 3)
            # random.choice(10*[0] + [3])

            self.move_car(random_action)

            if self.check_collision() or self.check_parking():
                running = False

            self.draw_environment()
            pygame.display.flip()

            CLOCK.tick(FPS)

        pygame.quit()
        sys.exit()


if __name__ == "__main__":
    test_env = Environment()
    test_env.test_run(60)
