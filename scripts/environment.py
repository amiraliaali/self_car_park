import pygame
import math
import sys
import time
import random
import cv2 as cv
import numpy as np

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
    "collision": -2000,
    "parked": 5000,
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
FPS = 120

CAR_WIDTH, CAR_HEIGHT = 50, 30


class Environment:
    def __init__(self) -> None:
        self.generate_car()
        self.generate_obstacle_cars()
        self.iteration_num = 0
        self.car_acceleration = 0.2
        self.car_friction = 0.05
        self.max_speed = 5
        self.steering_angle = 7
        self.parking_spot_x, self.parking_spot_y = 300, 300
        self.window_opened = False
        self.action_num = 0
        self.parked_tolerance_margin = 10
        self.env_reset()
        self.all_actions_current_run = list()

    def generate_car(self):
        self.car_surface = pygame.Surface((CAR_WIDTH, CAR_HEIGHT), pygame.SRCALPHA)
        self.car_surface.fill(BLACK)

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
        self.car_x, self.car_y = 50, 400
        self.car_angle = 0
        self.car_speed = 0
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
            (self.parking_spot_x, self.parking_spot_y, CAR_WIDTH, CAR_HEIGHT),
            2,
        )
        for obstacle in self.obstacle_cars:
            pygame.draw.rect(self.screen, RED, obstacle)

        rotated_car = pygame.transform.rotate(self.car_surface, -self.car_angle)
        car_rect = rotated_car.get_rect(center=(self.car_x, self.car_y))
        self.screen.blit(rotated_car, car_rect.topleft)

    def move_car(self, action):
        if action in [0, 1, 2]:  # Accelerate
            self.car_speed = min(self.max_speed, self.car_speed + self.car_acceleration)
        elif action in [3, 4, 5]:  # Decelerate
            self.car_speed = max(-self.max_speed, self.car_speed - self.car_acceleration)

        if action in [1, 4]:  # Left
            self.car_angle += self.steering_angle
        elif action in [2, 5]:  # Right
            self.car_angle -= self.steering_angle

        self.car_speed *= 1 - self.car_friction
        rad = math.radians(self.car_angle)
        self.car_x += self.car_speed * math.cos(rad)
        self.car_y += self.car_speed * math.sin(rad)


    def check_collision(self):
        """Check for collisions with obstacles or boundaries."""
        car_rect = pygame.Rect(
            self.car_x - CAR_WIDTH // 2,
            self.car_y - CAR_HEIGHT // 2,
            CAR_WIDTH,
            CAR_HEIGHT,
        )
        if (
            self.car_x < 0
            or self.car_x > WIDTH
            or self.car_y < 0
            or self.car_y > HEIGHT
        ):
            return True
        for obstacle in self.obstacle_cars:
            if car_rect.colliderect(obstacle):
                return True
        return False

    def check_parking(self):
        """Check if the car is parked in the designated spot and centered inside."""
        car_rect = pygame.Rect(
            self.car_x - CAR_WIDTH // 2,
            self.car_y - CAR_HEIGHT // 2,
            CAR_WIDTH,
            CAR_HEIGHT,
        )

        parking_rect = pygame.Rect(
            self.parking_spot_x, self.parking_spot_y, CAR_WIDTH, CAR_HEIGHT
        )

        # Check if the car's center is inside the parking spot
        car_center = car_rect.center
        parking_center = parking_rect.center

        # Check if the car's center is within the parking spot's center with some margin
        if (
            abs(car_center[0] - parking_center[0]) <= self.parked_tolerance_margin
            and abs(car_center[1] - parking_center[1]) <= self.parked_tolerance_margin
        ):
            return True
        return False

    def check_getting_closer(self):
        distance_to_parking_spot = math.sqrt(
            (self.car_x - self.parking_spot_x) ** 2
            + (self.car_y - self.parking_spot_y) ** 2
        )
        if distance_to_parking_spot < self.current_car_parking_distance:
            self.current_car_parking_distance = distance_to_parking_spot
            return True
        return False

    def calculate_distance_reward(self):
        distance_to_parking_spot = math.sqrt((self.car_x - self.parking_spot_x)**2 + (self.car_y - self.parking_spot_y)**2)
        max_distance = math.sqrt(WIDTH**2 + HEIGHT**2)
        distance_reward = 1 - (distance_to_parking_spot / max_distance)

        angle_to_parking_spot = math.atan2(self.parking_spot_y - self.car_y, self.parking_spot_x - self.car_x)
        angle_diff = abs(math.degrees(angle_to_parking_spot) - self.car_angle) % 360
        angle_reward = 1 - (angle_diff / 180)

        reward = distance_reward + 0.5 * angle_reward
        return reward


    def generate_video_current_run(self):
        self.env_reset(reset_actions_list = False)
        frame_width, frame_height = WIDTH, HEIGHT
        fourcc = cv.VideoWriter_fourcc(*'H264')  # Codec for MP4
        out = cv.VideoWriter(f'output_video_{self.iteration_num}.mp4', fourcc, FPS, (frame_width, frame_height))

        for i in self.all_actions_current_run:
            state = self.get_current_state()
            action = i
            self.move_car(action)

            # Draw environment in pygame
            self.draw_environment()
            pygame.display.flip()

            # Capture the current screen for the video
            frame = pygame.surfarray.array3d(pygame.display.get_surface())
            frame = np.transpose(frame, (1, 0, 2))  # (width, height, channels) -> (height, width, channels)
            frame = cv.cvtColor(frame, cv.COLOR_RGB2BGR)  # Convert RGB to BGR for OpenCV
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

        reward = self.calculate_distance_reward()

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

        if self.total_moves > 3000:
            self.total_moves = 0
            return True, REWARDS["time_up"], state
        
        # Default penalty for no progress
        return False, reward, state

    def get_current_state(self):
        return [
            (self.car_x - self.parking_spot_x) / WIDTH,
            (self.car_y - self.parking_spot_y) / HEIGHT,
            math.sqrt(
                (self.car_x - self.parking_spot_x) ** 2
                + (self.car_y - self.parking_spot_y) ** 2
            )
            / math.sqrt(WIDTH**2 + HEIGHT**2),
            (
                math.atan2(
                    self.parking_spot_y - self.car_y, self.parking_spot_x - self.car_x
                )
                - math.radians(self.car_angle)
            )
            / math.pi,  # Angle difference normalized
            self.car_speed / self.max_speed,
            self.car_angle / 360,
            # Add other features as needed
        ]

    def test_run(self, run_time_in_sec):
        self.env_reset()

        running = True
        start_time = time.time()

        while running:
            if time.time() - start_time > run_time_in_sec:
                running = False

            random_action = random.randint(0, 3)

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
