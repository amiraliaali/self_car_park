import pygame
import math
import sys

ACTIONS_MAPPING = {
    "0": "left",
    "1": "right",
    "2": "up",
    "3": "down"
}

REWARDS = {
    "collision": -200,
    "parked": 200,
    "getting_closer": 1,
    "time_up": -500,
    "else": -1
}

ACTION_KEY_MAPPING = {
        0: pygame.K_LEFT,
        1: pygame.K_RIGHT,
        2: pygame.K_UP,
        3: pygame.K_DOWN
    }

pygame.init()

WIDTH, HEIGHT = 800, 600

ACTION_NUM = 0

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY = (200, 200, 200)
RED = (255, 0, 0)
GREEN = (0, 255, 0)

clock = pygame.time.Clock()
FPS = 120

CAR_WIDTH, CAR_HEIGHT = 50, 30

# Car parameters
car_x, car_y = 100, 500
car_angle = 0
car_speed = 0
car_acceleration = 0.2
car_friction = 0.05
max_speed = 5
steering_angle = 3

parking_spot_x, parking_spot_y = 400, 300

current_car_parking_distance = math.inf

# Obstacle cars
obstacle_cars = [
    pygame.Rect(400, 400, CAR_WIDTH, CAR_HEIGHT),
    pygame.Rect(400, 350, CAR_WIDTH, CAR_HEIGHT),
    pygame.Rect(400, 250, CAR_WIDTH, CAR_HEIGHT),
    pygame.Rect(400, 200, CAR_WIDTH, CAR_HEIGHT),
]

car_surface = pygame.Surface((CAR_WIDTH, CAR_HEIGHT), pygame.SRCALPHA)
car_surface.fill(BLACK)

WINDOW_OPENED = False

def generate_window():
    global screen
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Parallel Parking Simulator")

def draw_environment():
    global WINDOW_OPENED, screen
    if not WINDOW_OPENED:
        generate_window()
        WINDOW_OPENED = True

    screen.fill(GRAY)
    pygame.draw.rect(screen, GREEN, (parking_spot_x, parking_spot_y, CAR_WIDTH, CAR_HEIGHT), 2)
    for obstacle in obstacle_cars:
        pygame.draw.rect(screen, RED, obstacle)

    rotated_car = pygame.transform.rotate(car_surface, -car_angle)
    car_rect = rotated_car.get_rect(center=(car_x, car_y))
    screen.blit(rotated_car, car_rect.topleft)

def move_car(action):
    global car_x, car_y, car_angle, car_speed

    # Apply acceleration/braking
    if action == 2:  # Accelerate
        car_speed = min(max_speed, car_speed + car_acceleration)
    elif action == 3:  # Brake
        car_speed = max(-max_speed, car_speed - car_acceleration)
    
    # Apply steering
    if action == 0:  # Steer left
        car_angle += steering_angle if car_speed != 0 else 0
    elif action == 1:  # Steer right
        car_angle -= steering_angle if car_speed != 0 else 0

    # Apply friction
    car_speed *= (1 - car_friction)

    # Update position
    rad = math.radians(car_angle)
    car_x += car_speed * math.cos(rad)
    car_y += car_speed * math.sin(rad)


def check_collision():
    """Check for collisions with obstacles or boundaries."""
    car_rect = pygame.Rect(car_x - CAR_WIDTH // 2, car_y - CAR_HEIGHT // 2, CAR_WIDTH, CAR_HEIGHT)
    if car_x < 0 or car_x > WIDTH or car_y < 0 or car_y > HEIGHT:
        return True
    for obstacle in obstacle_cars:
        if car_rect.colliderect(obstacle):
            return True
    return False

def check_parking():
    """Check if the car is parked in the designated spot and centered inside."""
    car_rect = pygame.Rect(car_x - CAR_WIDTH // 2, car_y - CAR_HEIGHT // 2, CAR_WIDTH, CAR_HEIGHT)
    
    # Define parking spot rectangle
    parking_rect = pygame.Rect(parking_spot_x, parking_spot_y, CAR_WIDTH, CAR_HEIGHT)

    # Check if the car's center is inside the parking spot
    car_center = car_rect.center
    parking_center = parking_rect.center

    # You can define a margin if you want a little tolerance for perfect parking
    margin = 5  # Pixel margin for tolerance

    # Check if the car's center is within the parking spot's center with some margin
    if (abs(car_center[0] - parking_center[0]) <= margin and
        abs(car_center[1] - parking_center[1]) <= margin):
        return True
    return False


def check_getting_closer():
    global current_car_parking_distance
    distance_to_parking_spot = math.sqrt((car_x - parking_spot_x)**2 + (car_y - parking_spot_y)**2)
    if distance_to_parking_spot < current_car_parking_distance:
        current_car_parking_distance = distance_to_parking_spot
        return True
    return False


def check_time_up():
    global ACTION_NUM
    return( ACTION_NUM > 5000)


def execute_move(action):
    """Execute the given action, return (done, reward, next_state)."""
    global ACTION_NUM, current_car_parking_distance

    # Update the car's state based on the action
    move_car(action)

    # Get the updated state
    state = get_current_state()

    # Increment action counter
    ACTION_NUM += 1

    # Check if time limit is reached
    # if check_time_up():
    #     return True, REWARDS["time_up"], state

    # Collision detection
    if check_collision():
        return True, REWARDS["collision"], state

    # Parking success
    if check_parking():
        return True, REWARDS["parked"], state

    # Reward for getting closer to the parking spot
    if check_getting_closer():
        return False, REWARDS["getting_closer"], state

    # Default penalty for no progress
    return False, REWARDS["else"], state



def get_current_state():
    distance_to_parking_spot = math.sqrt((car_x - parking_spot_x)**2 + (car_y - parking_spot_y)**2)
    angle_to_parking_spot = math.degrees(math.atan2(parking_spot_y - car_y, parking_spot_x - car_x)) - car_angle
    max_distance = math.sqrt(WIDTH**2 + HEIGHT**2)
    distance_to_parking_spot /= max_distance
    angle_to_parking_spot /= 360

    return [distance_to_parking_spot, angle_to_parking_spot, car_speed]


def reset():
    global car_x, car_y, car_angle, car_speed, current_car_parking_distance, ACTION_NUM
    car_x, car_y = 100, 500
    car_angle = 0
    car_speed = 0
    current_car_parking_distance = math.inf
    ACTION_NUM = 0
    pygame.event.clear()


def test_run():
    # Main game loop
    running = True
    while running:
        keys = pygame.key.get_pressed()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        move_car(keys)

        if check_collision():
            print("Collision! Resetting car position...")
            car_x, car_y, car_angle, car_speed = 100, 500, 0, 0

        if check_parking():
            print("Car parked successfully!")
            running = False

        # Draw everything
        draw_environment()
        pygame.display.flip()
        clock.tick(FPS)

    pygame.quit()
    sys.exit()
