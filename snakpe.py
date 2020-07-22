import pygame
import numpy as np


class Direction:
    DOWN = 0
    RIGHT = 1
    UP = 2
    LEFT = 3


class Node:
    def __init__(self, pos, next):
        self.pos = pos
        self.next = next


class Snake:
    def __init__(self, grid_max_x, grid_max_y):
        self.length = 1
        self.colour = (255, 255, 255)
        self.head = Node([0, 0], None)
        self.tail = Node([0, 0], self.head)
        self.direction = Direction.DOWN
        self.grid = np.zeros((grid_max_x, grid_max_y), dtype=int)
        self.grid[0][0] = 1
        self.max_x = grid_max_x
        self.max_y = grid_max_y
        self.apple = generate_apple(self.grid)
        self.done = False

        self.obs_space_n = grid_max_x * grid_max_y + 6
        self.action_space_n = 5

    def move(self):
        # Returns true if we land on an apple and do not shrink the tail
        # Returns false if we lose the game
        new_head_pos = list(self.head.pos)
        if self.direction == Direction.DOWN:
            new_head_pos[1] += 1
        elif self.direction == Direction.UP:
            new_head_pos[1] += -1
        elif self.direction == Direction.LEFT:
            new_head_pos[0] += -1
        elif self.direction == Direction.RIGHT:
            new_head_pos[0] += 1

        # Check if the game is lost
        if new_head_pos[0] < 0 or new_head_pos[0] >= self.max_x or new_head_pos[1] < 0 or new_head_pos[1] >= self.max_y:
            self.done = True
            return False

        # Check if we landed on an apple
        if new_head_pos[0] == self.apple[0] and new_head_pos[1] == self.apple[1]:
            self.length += 1
            new_head = Node(new_head_pos, None)
            self.head.next = new_head
            if self.length == 2:
                self.tail = self.head
            self.head = new_head
            self.grid[new_head_pos[0]][new_head_pos[1]] = 1
            self.apple = generate_apple(self.grid)
            return True

        # Check if the new head position is valid
        self.grid[self.tail.pos[0]][self.tail.pos[1]] = 0
        if self.grid[new_head_pos[0]][new_head_pos[1]] == 1:
            self.done = True
            return False

        self.grid[new_head_pos[0]][new_head_pos[1]] = 1
        new_head = Node(new_head_pos, None)
        self.head.next = new_head
        self.head = new_head

        old_tail = self.tail.pos
        if self.length == 1:
            self.tail = self.head
        else:
            self.tail = self.tail.next

        return old_tail

    def get_game_state(self):
        self.grid[self.apple[0]][self.apple[1]] = -1
        game_state = self.grid.flatten()
        self.grid[self.apple[0]][self.apple[1]] = 0
        game_summary = np.array([self.head.pos[0], self.head.pos[1], self.apple[0],
                                 self.apple[1], self.direction, self.length])
        return np.concatenate((game_state, game_summary), axis=None).reshape(-1, self.obs_space_n), self.done

    def reset(self):
        self.length = 1
        self.head = Node([0, 0], None)
        self.tail = Node([0, 0], self.head)
        self.direction = Direction.DOWN
        self.grid = np.zeros((self.max_x, self.max_y), dtype=int)
        self.grid[0][0] = 1
        self.apple = generate_apple(self.grid)
        self.done = False


def generate_apple(grid):
    # Returns a location in the grid where there is a 0 randomly
    empty = np.argwhere(grid == 0)
    idx = np.random.randint(len(empty))
    return empty[idx]


def render_snake(surface, snake, old_tail=None):
    if old_tail is not None:
        pygame.draw.rect(surface,
                         (0, 0, 0),
                         (old_tail[0] * cell_size, old_tail[1] * cell_size, cell_size, cell_size),
                         0)
    pygame.draw.rect(surface,
                     snake.colour,
                     (snake.head.pos[0]*cell_size, snake.head.pos[1]*cell_size, cell_size, cell_size),
                     0)
    pygame.draw.rect(surface,
                     (255, 0, 0),
                     (snake.apple[0] * cell_size, snake.apple[1] * cell_size, cell_size, cell_size),
                     0)
    # Drawing the grid cells
    # Vertical lines
    for i in range(screen_height // cell_size):
        pygame.draw.line(surface,
                         (128, 128, 128),
                         (0, i * cell_size),
                         (screen_height, i * cell_size))
    # Horizontal lines
    for i in range(screen_width // cell_size):
        pygame.draw.line(surface,
                         (128, 128, 128),
                         (i * cell_size, 0),
                         (i * cell_size, screen_width))
    pygame.display.update()


def display_text(surface, text, colour):
    font = pygame.font.SysFont('comicsans', 60, bold=True)
    to_display = font.render(text, 1, colour)
    surface.blit(to_display,
                 (screen_width/2 - text.get_width()/2,
                  screen_height/2 - text.get_height()/2))
    pygame.display.update()


def main():
    run = True
    clock = pygame.time.Clock()
    time_elapsed = 0
    tick_time = 0.2

    snake = Snake(screen_height//cell_size, screen_width//cell_size)
    render_snake(screen, snake)

    while run:
        time_elapsed += clock.get_rawtime()
        clock.tick()

        if time_elapsed/1000 >= tick_time:
            time_elapsed = 0
            moved_snake = snake.move()
            if moved_snake is True:
                render_snake(screen, snake)
                if snake.length == snake.max_x * snake.max_y:
                    display_text(screen, "YOU WIN", (0, 255, 0))
                    pygame.time.delay(6000)
                    break
            elif moved_snake is False:
                display_text(screen, "YOU LOSE", (200, 0, 0))
                pygame.time.delay(6000)
                break
            else:
                render_snake(screen, snake, old_tail=moved_snake)

        for event in pygame.event.get():
            # Handle quitting
            if event.type == pygame.QUIT:
                run = False
                pygame.display.quit()
                quit()
            # Handle key inputs
            if event.type == pygame.KEYDOWN:
                keys = pygame.key.get_pressed()
                if keys[pygame.K_DOWN]:
                    if snake.direction != Direction.UP:
                        snake.direction = Direction.DOWN
                elif keys[pygame.K_RIGHT]:
                    if snake.direction != Direction.LEFT:
                        snake.direction = Direction.RIGHT
                elif keys[pygame.K_UP]:
                    if snake.direction != Direction.DOWN:
                        snake.direction = Direction.UP
                elif keys[pygame.K_LEFT]:
                    if snake.direction != Direction.RIGHT:
                        snake.direction = Direction.LEFT


if __name__ == "__main__":
    # Game configurations
    screen_width = 300
    screen_height = 300
    cell_size = 30
    assert screen_width % cell_size == 0 and screen_height % cell_size == 0

    pygame.font.init()
    screen = pygame.display.set_mode((screen_width, screen_height))
    pygame.display.set_caption('Snakpe')
    pygame.key.set_repeat(50)
    screen.fill((0, 0, 0))

    main()
