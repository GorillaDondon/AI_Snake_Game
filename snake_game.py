# In this game, a snake catches the maximum number of fruits without hitting the wall or itself
# AI takes care of which ways for snakes to go

# importing libraries
import pygame
import time
import random
import math

class Cell():
    def __init__(self, position, orientation, score = None):
        self.position = position
        self.orientation = orientation
        self.score = score
        
    def get_position(self):
        return self.position
    
    def set_position(self, position):
        self.position = position
    
    def get_score(self):
        return self.score

    def set_score(self, score):
        self.score = score

    def get_orientation(self):
        return self.orientation

def cell_score(cell, fruit_cell, window_x, window_y):
    # calculate the straight distance between a cell and the fruit
    distance = math.sqrt((fruit_cell[0] - cell[0])**2 + (fruit_cell[1] - cell[1])**2)
    if distance == 0:
        return 1

    
    # when really close to the wall (i.g. within 3 units of the wall, we can significantly decrease the wall_safety_score )

    # calculate how likely the snake hits itself
    """

    """

    # calculate the score 
    #score = distance

    #return score
    return 1 / distance

def get_best_cell(cell_up, cell_right, cell_down, cell_left, fruit_position, window_x, window_y):
    cell_up.set_score(cell_score(cell_up.get_position(), fruit_position, window_x, window_y))
    cell_right.set_score(cell_score(cell_right.get_position(), fruit_position, window_x, window_y))
    cell_down.set_score(cell_score(cell_down.get_position(), fruit_position, window_x, window_y))
    cell_left.set_score(cell_score(cell_left.get_position(), fruit_position, window_x, window_y))

    best_cell = max([cell_up, cell_right, cell_down, cell_left], key=lambda cell:cell.score)

    return best_cell

# displaying Score function
def show_score(choice, color, font, size, game_window, score):
    # creating font object score_font
    score_font = pygame.font.SysFont(font, size)
    
    # create the display surface object 
    # score_surface
    score_surface = score_font.render('Score : ' + str(score), True, color)
    
    # create a rectangular object for the text
    # surface object
    score_rect = score_surface.get_rect()
    
    # displaying text
    game_window.blit(score_surface, score_rect)

# game over function
def game_over(game_window, score, window_x, window_y, red):
  
    # creating font object my_font
    my_font = pygame.font.SysFont('times new roman', 50)
    
    # creating a text surface on which text 
    # will be drawn
    game_over_surface = my_font.render(
        'Your Score is : ' + str(score), True, red)
    
    # create a rectangular object for the text 
    # surface object
    game_over_rect = game_over_surface.get_rect()
    
    # setting position of the text
    game_over_rect.midtop = (window_x/2, window_y/4)
    
    # blit will draw the text on screen
    game_window.blit(game_over_surface, game_over_rect)
    pygame.display.flip()
    
    # after 2 seconds we will quit the program
    time.sleep(2)
    
    # deactivating pygame library
    pygame.quit()
    
    # quit the program
    quit()

def game():
    snake_speed = 10

    # Window size
    window_x = 700
    window_y = 500

    # defining colors
    black = pygame.Color(0, 0, 0)
    white = pygame.Color(255, 255, 255)
    red = pygame.Color(255, 0, 0)
    green = pygame.Color(0, 255, 0)
    blue = pygame.Color(0, 0, 255)

    # Initialising pygame
    pygame.init()

    # Initialise game window
    pygame.display.set_caption('Welcome to the Snake Game 480')
    game_window = pygame.display.set_mode((window_x, window_y))

    # FPS (frames per second) controller
    fps = pygame.time.Clock()

    # defining snake default position
    snake_position = [100, 50]

    # defining first 4 blocks of snake body
    snake_body = [[100, 50],
                [90, 50],
                [80, 50],
                [70, 50]
                ]
    # fruit position
    fruit_position = [random.randrange(1, (window_x//10)) * 10, 
                    random.randrange(1, (window_y//10)) * 10]

    fruit_spawn = True

    # setting default snake direction towards
    # right
    direction = 'RIGHT'
    change_to = direction

    # initial score
    score = 0

    # Main Function
    while True:
        # handling key events
        for event in pygame.event.get():
            pass

        cell_up = Cell((snake_position[0], snake_position[1] - 10), 'UP')
        cell_right = Cell((snake_position[0] + 10, snake_position[1]), 'RIGHT')
        cell_down = Cell((snake_position[0], snake_position[1] + 10), 'DOWN')
        cell_left = Cell((snake_position[0] - 10, snake_position[1]), 'LEFT')

        best_cell = get_best_cell(cell_up, cell_right, cell_down, cell_left, fruit_position, window_x, window_y)

        change_to = best_cell.get_orientation()
        
        # If two keys pressed simultaneously
        # we don't want snake to move into two 
        # directions simultaneously
        if change_to == 'UP' and direction != 'DOWN':
            direction = 'UP'
        if change_to == 'DOWN' and direction != 'UP':
            direction = 'DOWN'
        if change_to == 'LEFT' and direction != 'RIGHT':
            direction = 'LEFT'
        if change_to == 'RIGHT' and direction != 'LEFT':
            direction = 'RIGHT'

        # Moving the snake
        if direction == 'UP':
            snake_position[1] -= 10
        if direction == 'DOWN':
            snake_position[1] += 10
        if direction == 'LEFT':
            snake_position[0] -= 10
        if direction == 'RIGHT':
            snake_position[0] += 10

        print(direction)

        # Snake body growing mechanism
        # if fruits and snakes collide then scores
        # will be incremented by 10
        snake_body.insert(0, list(snake_position)) # snake making forward movement
        if snake_position[0] == fruit_position[0] and snake_position[1] == fruit_position[1]:
            score += 10
            fruit_spawn = False
        else:
            snake_body.pop()
            
        if not fruit_spawn:
            fruit_position = [random.randrange(1, (window_x//10)) * 10, 
                            random.randrange(1, (window_y//10)) * 10]
            
        fruit_spawn = True
        game_window.fill(black)
        
        for pos in snake_body:
            pygame.draw.rect(game_window, green,
                            pygame.Rect(pos[0], pos[1], 10, 10))
        pygame.draw.rect(game_window, white, pygame.Rect(
            fruit_position[0], fruit_position[1], 10, 10))

        # Game Over conditions
        if snake_position[0] < 0 or snake_position[0] > window_x-10:
            game_over(game_window, score, window_x, window_y, red)
        if snake_position[1] < 0 or snake_position[1] > window_y-10:
            game_over(game_window, score, window_x, window_y, red)

        # Touching the snake body
        for block in snake_body[1:]:
            if snake_position[0] == block[0] and snake_position[1] == block[1]:
                game_over(game_window, score, window_x, window_y, red)

        # displaying score continuously
        show_score(1, white, 'times new roman', 20, game_window, score)

        # Refresh game screen
        pygame.display.update()

        # Frame Per Second /Refresh Rate
        fps.tick(snake_speed)

game()