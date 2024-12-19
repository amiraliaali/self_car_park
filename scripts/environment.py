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
    "0": "accelerate",
    "1": "decelerate",
    "2": "right",
    "3": "left",
}

REWARDS = {
    "collision": -500,
    "parked": 1000,
    "time_up": -250,
}

pygame.init()

WIDTH, HEIGHT = 500, 400

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY = (200, 200, 200)
RED = (255, 0, 0)
GREEN = (0, 255, 0)

CLOCK = pygame.time.Clock()
FPS = 30


class Environment:
    def __init__(self) -> None:
        self.spots = [
            (400, 340),
            (400, 300),
            (400, 260),
            (400, 220),
            (400, 180),
            (400, 140),
            (400, 100),
            (260, 220),
            (260, 180),
            (260, 140),
            (260, 100),
            (200, 220),
            (200, 180),
            (200, 140),
            (200, 100),
        ]
        self.car_x, self.car_y = 50, 250
        self.generate_car()
        self.iteration_num = -1
        self.parking_spot_x, self.parking_spot_y = 400, 140
        self.window_opened = False
        self.action_num = 0
        self.parked_tolerance_margin = 5
        self.env_reset()
        self.all_actions_current_run = list()
        self.all_reward_current_run = list()
        self.all_angle_rewards = list()
        self.lidar_info = list()
        self.max_dist = math.sqrt(WIDTH**2 + HEIGHT**2)

    def generate_car(self):
        self.car_agent = Car(self.car_x, self.car_y)
        self.car_surface = pygame.Surface(
            (self.car_agent.width, self.car_agent.height), pygame.SRCALPHA
        )
        self.car_surface.fill(BLACK)

    def boundary_collision(self):
        """Check for collisions with boundaries."""
        if (
            self.car_agent.x < 0
            or self.car_agent.x > WIDTH
            or self.car_agent.y < 0
            or self.car_agent.y > HEIGHT
        ):
            return True
        return False

    def generate_obstacle_cars(self):
        self.obstacle_cars = [
            pygame.Rect(x, y, self.car_agent.width, self.car_agent.height)
            for (x, y) in self.obstacles
        ]

    def env_reset(self, generate_video=False):
        pygame.init()
        # self.generate_car(50, random.choice([100, 125, 150, 175, 200, 225, 250, 275, 300]))
        pygame.event.clear()
        if not generate_video:
            self.car_x = random.choice([50, 100, 350])
            self.car_y = random.choice([100, 150, 200, 250, 300])
            self.parking_spot_x, self.parking_spot_y = random.choice(self.spots)
            self.obstacles = self.spots.copy()
            self.obstacles.remove((self.parking_spot_x, self.parking_spot_y))
            self.generate_obstacle_cars()
            self.total_moves = 0
            self.all_actions_current_run = []
            self.all_reward_current_run = []
            self.all_angle_rewards = []
            self.lidar_info = []
            self.iteration_num += 1
        self.car_agent.reset(self.car_x, self.car_y)
        self.current_car_parking_distance = math.inf
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
        if self.boundary_collision():
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

    def calculate_distance_reward(self):
        """Calculates the reward based on the position in the gaussian plane."""
        x_diff = self.car_agent.x - self.parking_spot_x
        y_diff = self.car_agent.y - self.parking_spot_y
        dist = math.sqrt(x_diff**2 + y_diff**2)
        dist_normalized = dist / self.max_dist
        return 2 * math.exp(-dist_normalized)

    def calculate_angle_reward(self):
        # Compute vector differences
        dy = self.parking_spot_y - self.car_agent.y
        dx = self.parking_spot_x - self.car_agent.x

        # Compute angle to parking spot in degrees
        angle_to_parking_spot = math.degrees(math.atan2(dy, dx))

        # Normalize angles to [0, 360)
        car_angle = self.car_agent.angle % 360
        angle_to_parking_spot = angle_to_parking_spot % 360

        # Compute the angular difference and normalize to [0, 180]
        diff = abs(car_angle - angle_to_parking_spot)
        diff = min(diff, 360 - diff)

        # Compute reward: Higher reward for smaller angular difference
        reward = 1 - diff / 180
        return reward

    def generate_video_current_run(self):
        self.env_reset(True)
        frame_width, frame_height = WIDTH, HEIGHT
        fourcc = cv.VideoWriter_fourcc(*"H264")  # Codec for MP4
        out = cv.VideoWriter(
            f"output_videos/{self.iteration_num}.mp4",
            fourcc,
            FPS,
            (frame_width, frame_height),
        )

        for i in range(len(self.all_actions_current_run)):
            state = self.get_current_state()
            action = self.all_actions_current_run[i]
            self.move_car(action)
            distances = {}

            # Draw environment in pygame
            self.draw_environment()

            # Draw lidar lines using Pygame
            # if not i+1>(len(self.all_actions_current_run)):
            lidar_info = self.lidar_info[i+1]  # Get lidar information for the current frame
            for direction, info in lidar_info.items():
                start = info["starting_point"]
                end = info["ending_point"]
                dist = info["min_dist"]
                distances[direction] = dist

                pygame.draw.line(self.screen, (0, 0, 255), start, end, 1)

            pygame.display.flip()


            # Capture the current screen for the video
            frame = pygame.surfarray.array3d(pygame.display.get_surface())
            frame = np.transpose(
                frame, (1, 0, 2)
            )  # (width, height, channels) -> (height, width, channels)
            frame = cv.cvtColor(
                frame, cv.COLOR_RGB2BGR
            )  # Convert RGB to BGR for OpenCV

            # Get the reward for the current frame
            position_reward = self.all_reward_current_run[i]
            angle_reward = self.all_angle_rewards[i]
            total_reward = sum(self.all_reward_current_run[:i]) + sum(
                self.all_angle_rewards[:i]
            )

            # Add the reward text to the frame
            font = cv.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            color = (0, 0, 0)  # Green color for the reward text
            thickness = 2

            lidar_info_ys = 20
            for direction,dist in distances.items():
                cv.putText(frame, f"{direction.upper()[0]}: {dist:.2f}", (lidar_info_ys, 75), font, font_scale, color, 1)
                lidar_info_ys += 120


            text = f"Pos. Rew.: {position_reward:.2f}"  # Format reward to 2 decimal places

            cv.putText(frame, text, (20, 50), font, font_scale, color, 1)

            text = (
                f"Angle Rew.: {angle_reward:.2f}"  # Format reward to 2 decimal places
            )

            cv.putText(frame, text, (190, 50), font, font_scale, color, 1)

            text = (
                f"Iteration: {self.iteration_num}"  # Format reward to 2 decimal places
            )

            cv.putText(frame, text, (20, 25), font, font_scale, color, thickness)

            text = (
                f"Tot. Rew.: {total_reward:.2f}"  # Format reward to 2 decimal places
            )

            cv.putText(frame, text, (350, 50), font, font_scale, color, 1)

            text = f"Car X: {self.car_agent.x:.2f}"  # Format reward to 2 decimal places

            cv.putText(frame, text, (160, 25), font, font_scale, color, 1)

            text = f"Car Y: {self.car_agent.y:.2f}"  # Format reward to 2 decimal places

            cv.putText(frame, text, (340, 25), font, font_scale, color, 1)

            out.write(frame)

            CLOCK.tick(FPS)
        print("Finished saving video!!")
        out.release()
        self.all_actions_current_run = []
        self.all_reward_current_run = []
        self.all_angle_rewards = []
        self.lidar_info = []
        self.screen.fill(GRAY)

    def raycast(self, start, angle, max_distance):
        x, y = start
        dx = math.cos(angle)
        dy = math.sin(angle)
        step_size = 1
        distance = 0

        while distance < max_distance:
            # Move the ray forward
            x += dx * step_size
            y += dy * step_size
            distance += step_size

            # Check if the ray hits any obstacle
            for obstacle in self.obstacle_cars:
                if obstacle.collidepoint(x, y):
                    return (x, y), distance  # Intersection point and distance

            # Stop if the ray moves out of bounds
            if x < 0 or x > WIDTH or y < 0 or y > HEIGHT:
                return (x, y), max_distance

        # If no collision, return max distance
        return (x, y), max_distance

    def four_directions_lidar(self):
        current_x = self.car_agent.x
        current_y = self.car_agent.y
        current_angle = math.radians(self.car_agent.angle)
        right_angle = math.radians(self.car_agent.angle+90)
        back_angle = math.radians(self.car_agent.angle+180)
        left_angle = math.radians(self.car_agent.angle-90)
        half_width = self.car_agent.width / 2
        half_height = self.car_agent.height / 2

        max_dist = math.sqrt(WIDTH**2 + HEIGHT**2)

        min_distances = {"forward": max_dist, "right": max_dist, "back": max_dist, "left": max_dist}

        starting_points = {
            "forward": (
                current_x,
                current_y,
            ),
            "right": (
                current_x,
                current_y,
            ),
            "back": (
                current_x,
                current_y,
            ),
            "left": (
                current_x,
                current_y,
            ),
        }

        end_points = {
            "forward": (
                current_x + (half_width+700) * math.cos(current_angle),
                current_y + (half_height+700) * math.sin(current_angle),
            ),
            "right": (
                current_x + (half_width+700) * math.cos(right_angle),
                current_y + (half_height+700) * math.sin(right_angle),
            ),
            "back": (
                current_x + (half_width+700) * math.cos(back_angle),
                current_y + (half_height+700) * math.sin(back_angle),
            ),
            "left": (
                current_x + (half_width+700) * math.cos(left_angle),
                current_y + (half_height+700) * math.sin(left_angle),
            ),
        }

        for direction, s_point in starting_points.items():
            angle = current_angle
            if direction == "right":
                angle = right_angle
            elif direction == "back":
                angle = back_angle
            elif direction == "left":
                angle = left_angle
            end_point, dist = self.raycast(s_point, angle, max_dist)
            if min_distances[direction] > dist:
                min_distances[direction] = dist
                end_points[direction] = end_point

        lidar_info = {
            "forward": {
                "starting_point": starting_points["forward"],
                "ending_point": end_points["forward"],
                "min_dist": min_distances["forward"],
            },
            "right": {
                "starting_point": starting_points["right"],
                "ending_point": end_points["right"],
                "min_dist": min_distances["right"],
            },
            "back": {
                "starting_point": starting_points["back"],
                "ending_point": end_points["back"],
                "min_dist": min_distances["back"],
            },
            "left": {
                "starting_point": starting_points["left"],
                "ending_point": end_points["left"],
                "min_dist": min_distances["left"],
            }
        }

        self.lidar_info.append(lidar_info)
        return lidar_info

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
            self.all_reward_current_run.append(REWARDS["collision"])
            self.all_angle_rewards.append(self.calculate_angle_reward())
            if self.iteration_num % 500 == 0:
                self.generate_video_current_run()
            return True, REWARDS["collision"], state

        # Parking success
        if self.check_parking():
            print(f"parked succesfully. Reward: {REWARDS['parked']}")
            self.all_reward_current_run.append(REWARDS["parked"])
            self.all_angle_rewards.append(self.calculate_angle_reward())
            if self.iteration_num % 1 == 0:
                self.generate_video_current_run()
            return True, REWARDS["parked"], state

        if self.total_moves > 1000:
            self.total_moves = 0
            self.all_reward_current_run.append(REWARDS["time_up"])
            self.all_angle_rewards.append(self.calculate_angle_reward())
            if self.iteration_num % 500 == 0:
                self.generate_video_current_run()
            return True, REWARDS["time_up"], state

        reward = self.calculate_distance_reward() + self.calculate_angle_reward()

        self.all_reward_current_run.append(self.calculate_distance_reward())
        self.all_angle_rewards.append(self.calculate_angle_reward())

        # Default penalty for no progress
        return False, reward, state

    def calc_car_distance(self, x, y):
        dy = self.car_agent.y - y
        dx = self.car_agent.x - x

        return math.sqrt(dy**2 + dx**2)

    def get_current_state(self):
        # Compute vector differences
        dy = self.parking_spot_y - self.car_agent.y
        dx = self.parking_spot_x - self.car_agent.x

        # Compute angle to parking spot in degrees
        angle_to_parking_spot = math.degrees(math.atan2(dy, dx))

        # Normalize angles to [0, 360)
        car_angle = self.car_agent.angle % 360
        angle_to_parking_spot = angle_to_parking_spot % 360

        diff_angle = abs(car_angle - angle_to_parking_spot)
        diff_angle = min(diff_angle, 360 - diff_angle) / 180

        distance_to_parking_spot = self.calc_car_distance(
            self.parking_spot_x, self.parking_spot_y
        )
        max_dist = self.calc_car_distance(WIDTH, HEIGHT)
        distances_to_obst = []
        for x, y in self.obstacles:
            if (x, y) != (self.parking_spot_x, self.parking_spot_y):
                distances_to_obst.append(self.calc_car_distance(x, y) / max_dist)

        obtalces_coordinates = []
        for x, y in self.obstacles:
            if (x, y) != (self.parking_spot_x, self.parking_spot_y):
                obtalces_coordinates.append(x / WIDTH)
                obtalces_coordinates.append(y / HEIGHT)
        
        four_sides_distances_dic = self.four_directions_lidar()
        four_sides_distances = []

        for direc in four_sides_distances_dic:
            four_sides_distances.append(four_sides_distances_dic[direc]["min_dist"] / max_dist)

        return [
            diff_angle,
            distance_to_parking_spot / max_dist,
            * four_sides_distances
            # *distances_to_obst,
            # self.car_agent.angle / 180,
            # self.car_agent.x / WIDTH,
            # self.car_agent.y / HEIGHT,
            # self.parking_spot_x / WIDTH,
            # self.parking_spot_y / HEIGHT,
            # *obtalces_coordinates
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
