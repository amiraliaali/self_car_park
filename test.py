import pygame
import math

# Initialize Pygame
pygame.init()

# Screen dimensions
WIDTH, HEIGHT = 800, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
clock = pygame.time.Clock()

# Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255, 0, 0)

# Obstacles (rectangles)
obstacles = [
    pygame.Rect(300, 200, 200, 100),
    pygame.Rect(500, 400, 100, 150)
]

def raycast(start, angle, max_distance, obstacles):
    x, y = start
    dx = math.cos(angle)  # Direction vector X
    dy = math.sin(angle)  # Direction vector Y
    step_size = 1  # Step size for ray movement
    distance = 0

    while distance < max_distance:
        # Move the ray forward
        x += dx * step_size
        y += dy * step_size
        distance += step_size

        # Check if the ray hits any obstacle
        for obstacle in obstacles:
            if obstacle.collidepoint(x, y):
                return (x, y), distance  # Intersection point and distance

        # Stop if the ray moves out of bounds
        if x < 0 or x > WIDTH or y < 0 or y > HEIGHT:
            break

    # If no collision, return max distance
    return (x, y), distance

# Main loop
running = True
while running:
    screen.fill(BLACK)

    # Draw obstacles
    for obstacle in obstacles:
        pygame.draw.rect(screen, WHITE, obstacle)

    # Get mouse position and shoot the ray
    start_pos = (WIDTH // 2, HEIGHT // 2)  # Start point of the ray
    mouse_pos = pygame.mouse.get_pos()
    angle = math.atan2(mouse_pos[1] - start_pos[1], mouse_pos[0] - start_pos[0])
    end_pos, distance = raycast(start_pos, angle, max_distance=1000, obstacles=obstacles)

    # Draw the ray
    pygame.draw.line(screen, RED, start_pos, end_pos, 2)

    # Display distance on the screen
    font = pygame.font.SysFont(None, 36)
    distance_text = font.render(f"Distance: {int(distance)}", True, WHITE)
    screen.blit(distance_text, (10, 10))

    # Event handling
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    pygame.display.flip()
    clock.tick(60)

pygame.quit()
