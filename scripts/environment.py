import pygame
import math
import sys
import time
import random

ACTIONS_MAPPING = {"0": "left", "1": "right", "2": "up", "3": "down"}

REWARDS = {
    "collision": -200,
    "parked": 200,
    "getting_closer": 1,
    "time_up": -500,
    "else": -1,
}

ACTION_KEY_MAPPING = {
    0: pygame.K_LEFT,
    1: pygame.K_RIGHT,
    2: pygame.K_UP,
    3: pygame.K_DOWN,
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

car_surface = pygame.Surface((CAR_WIDTH, CAR_HEIGHT), pygame.SRCALPHA)
car_surface.fill(BLACK)


class Environment:
    def __init__(self) -> None:
        self.env_reset()
        self.generate_obstacle_cars()
        self.car_acceleration = 0.2
        self.car_friction = 0.05
        self.max_speed = 5
        self.steering_angle = 3
        self.parking_spot_x, self.parking_spot_y = 400, 300
        self.window_opened = False
        self.action_num = 0
        self.parked_tolerance_margin = 5

    def generate_obstacle_cars(self):
        self.obstacle_cars = [
            pygame.Rect(400, 400, CAR_WIDTH, CAR_HEIGHT),
            pygame.Rect(400, 350, CAR_WIDTH, CAR_HEIGHT),
            pygame.Rect(400, 250, CAR_WIDTH, CAR_HEIGHT),
            pygame.Rect(400, 200, CAR_WIDTH, CAR_HEIGHT),
        ]

    def env_reset(self):
        self.car_x, self.car_y = 100, 500
        self.car_angle = 0
        self.car_speed = 0
        self.current_car_parking_distance = math.inf
        pygame.event.clear()

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

        rotated_car = pygame.transform.rotate(car_surface, -self.car_angle)
        car_rect = rotated_car.get_rect(center=(self.car_x, self.car_y))
        self.screen.blit(rotated_car, car_rect.topleft)

    def move_car(self, action):

        if action == 1:  # Accelerate
            self.car_speed = min(self.max_speed, self.car_speed + self.car_acceleration)
        elif action == 0:  # Brake
            self.car_speed = max(-self.max_speed, self.car_speed - self.car_acceleration)

        if action == 3:
            self.car_angle += self.steering_angle if self.car_speed != 0 else 0
        elif action == 4:
            self.car_angle -= self.steering_angle if self.car_speed != 0 else 0

        # Apply friction
        self.car_speed *= 1 - self.car_friction

        # Update position
        rad = math.radians(self.car_angle)
        self.car_x += self.car_speed * math.cos(rad)
        self.car_y += self.car_speed * math.sin(rad)

    def check_collision(self):
        """Check for collisions with obstacles or boundaries."""
        car_rect = pygame.Rect(
            self.car_x - CAR_WIDTH // 2, self.car_y - CAR_HEIGHT // 2, CAR_WIDTH, CAR_HEIGHT
        )
        if self.car_x < 0 or self.car_x > WIDTH or self.car_y < 0 or self.car_y > HEIGHT:
            return True
        for obstacle in self.obstacle_cars:
            if car_rect.colliderect(obstacle):
                return True
        return False

    def check_parking(self):
        """Check if the car is parked in the designated spot and centered inside."""
        car_rect = pygame.Rect(
            self.car_x - CAR_WIDTH // 2, self.car_y - CAR_HEIGHT // 2, CAR_WIDTH, CAR_HEIGHT
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
            (self.car_x - self.parking_spot_x) ** 2 + (self.car_y - self.parking_spot_y) ** 2
        )
        if distance_to_parking_spot < current_car_parking_distance:
            current_car_parking_distance = distance_to_parking_spot
            return True
        return False

    def execute_move(self, action):
        """Execute the given action, return (done, reward, next_state)."""
        # Update the car's state based on the action
        self.move_car(action)

        # Get the updated state
        state = self.get_current_state()

        # Collision detection
        if self.check_collision():
            return True, REWARDS["collision"], state

        # Parking success
        if self.check_parking():
            return True, REWARDS["parked"], state

        # Reward for getting closer to the parking spot
        if self.check_getting_closer():
            return False, REWARDS["getting_closer"], state

        # Default penalty for no progress
        return False, REWARDS["else"], state

    def get_current_state(self):
        distance_to_parking_spot = math.sqrt(
            (self.car_x - self.parking_spot_x) ** 2 + (self.car_y - self.parking_spot_y) ** 2
        )
        angle_to_parking_spot = (
            math.degrees(
                math.atan2(self.parking_spot_y - self.car_y, self.parking_spot_x - self.car_x)
            )
            - self.car_angle
        )
        max_distance = math.sqrt(WIDTH**2 + HEIGHT**2)
        distance_to_parking_spot /= max_distance
        angle_to_parking_spot /= 360

        return [distance_to_parking_spot, angle_to_parking_spot, self.car_speed]

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
