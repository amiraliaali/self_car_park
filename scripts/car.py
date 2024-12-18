import math
import pygame
import random

CAR_WIDTH, CAR_HEIGHT = 40, 20

WIDTH, HEIGHT = 500, 400


class Car:
    def __init__(self, x, y):
        self.width = CAR_WIDTH
        self.height = CAR_HEIGHT
        self.x = x + self.width // 2
        self.y = y + self.height // 2
        self.speed = 0
        self.acceleration = 0.2
        self.angle = 0
        self.fricition = 0.01
        self.max_speed = 5
        self.min_speed = -2
        self.steering_angle = 10
        self.initial_x = x
        self.initial_y = y

    def reset(self, x, y):
        """Reset the car's position."""
        # self.x = self.initial_x + self.width // 2
        # self.y = WIDTH - self.initial_y + self.height // 2
        self.x = x
        self.y = y

        self.speed = 0
        self.angle = 0

    def reset_randomly(self):
        self.x = random.randrange(50, 250, 20)
        self.y = random.randrange(100, 350, 20)

        self.speed = 0
        self.angle = 0

    def increase_speed(self):
        self.speed += self.acceleration
        self.speed = min(self.speed, self.max_speed)

    def decrease_speed(self):
        self.speed -= self.acceleration
        self.speed = max(self.speed, self.min_speed)

    def steer_left(self):
        self.angle += self.steering_angle
        self.angle %= 360
    
    def steer_right(self):
        self.angle -= self.steering_angle
        self.angle %= 360

    def update(self):
        """Update the car's position based on speed and angle."""

        if self.speed > 0:
            self.speed = max(0, self.speed - self.fricition)    

        elif self.speed < 0:
            self.speed = min(0, self.speed + self.fricition)

        self.speed = self.speed + self.acceleration
        if self.speed > self.max_speed:
            self.speed = self.max_speed
        elif self.speed < self.min_speed:
            self.speed = self.min_speed

        rad = math.radians(self.angle)
        self.x += self.speed * math.cos(rad)
        self.y += self.speed * math.sin(rad)
    
    def check_parking(self, parking_spot_x, parking_spot_y, parked_tolerance_margin):
        """Check if the car is parked in the designated spot and centered inside."""
        car_rect = pygame.Rect(
            self.x - self.width // 2,
            self.y - self.height // 2,
            self.width,
            self.height,
        )

        parking_rect = pygame.Rect(
            parking_spot_x, parking_spot_y, self.width, self.height
        )

        # Check if the car's center is inside the parking spot
        car_center = car_rect.center
        parking_center = parking_rect.center

        # Check if the car's center is within the parking spot's center with some margin
        if (
            abs(car_center[0] - parking_center[0]) <= parked_tolerance_margin
            and abs(car_center[1] - parking_center[1]) <= parked_tolerance_margin
        ):
            return True
        return False
    
    def distance_to_parking(self, parking_spot_x, parking_spot_y):
        """Calculate the distance to the parking spot."""
        distance_to_parking_spot = math.sqrt(
            (self.x - parking_spot_x) ** 2 + (self.y - parking_spot_y) ** 2
        )
        return distance_to_parking_spot
    
    def angle_to_parking(self, parking_spot_x, parking_spot_y):
        """Calculate the angle to the parking spot."""
        angle_to_parking_spot = math.atan2(parking_spot_y - self.y, parking_spot_x - self.x)
        return math.degrees(angle_to_parking_spot)

        
