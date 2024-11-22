# importing libraries
import pygame
import time
import random
import numpy as np
import os

# Snake game settings
snake_speed = 500  # Speed of the snake
window_x = 720
window_y = 480
population_size = 50  # Number of neural networks in each generation
initial_mutation_rate = 0.1  # Initial probability of mutation
num_generations = 50  # Number of generations
visualize_count = 3  # Number of neural networks to visualize from each generation


black = pygame.Color(0, 0, 0)
white = pygame.Color(255, 255, 255)
red = pygame.Color(255, 0, 0)
green = pygame.Color(0, 255, 0)
blue = pygame.Color(0, 0, 255)

pygame.init()

# Initialise game window
pygame.display.set_caption('Snake AI Game with Genetic Algorithm')
game_window = pygame.display.set_mode((window_x, window_y))
fps = pygame.time.Clock()


# Simple random initialization for neural network weights
def random_initialization(layer_size, prev_layer_size):
    return np.random.randn(layer_size, prev_layer_size) * 0.01  # Small random values

# Neural Network class
# used for each snake AI generation
# will need modification for inputs and layers as we progress
class NeuralNetwork:
    def __init__(self, input_size=14, hidden_layers=[16, 32, 64, 128, 256], output_size=4):
        # Neural Network Structure
        # Input size represents # of inputs given to the neural network using the get_game_state function
        # Hidden layer values represent the number of neurons in a hidden layer, optimizable
        # Output size represents the 4 directions the neural network can possibly output as a direction for the snake
        # to go to

        self.weights = []
        self.biases = []

        prev_layer_size = input_size

        # Generates the weights for each layer, connecting the neurons from layer to layer
        for hidden_size in hidden_layers:
            self.weights.append(random_initialization(hidden_size, prev_layer_size))
            self.biases.append(np.zeros((hidden_size, 1)))
            prev_layer_size = hidden_size


        self.weights.append(random_initialization(output_size, prev_layer_size))
        self.biases.append(np.zeros((output_size, 1)))

        # Momentum is used to accelerate the learning process of the model to push model towards optimization
        # Variance represents how much a model's output can fluctuate, specifically related to
        # overfitting/underfitting issues
        self.momentum_weights = [np.zeros_like(w) for w in self.weights]
        self.variance_weights = [np.zeros_like(w) for w in self.weights]
        self.momentum_biases = [np.zeros_like(b) for b in self.biases]
        self.variance_biases = [np.zeros_like(b) for b in self.biases]

    # Relu and Sigmoid functions used to squash outputs
    def relu(self, x):
        return np.maximum(0, x)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def forward(self, inputs):
        #Propagates the input information forward to the other hidden layers, and hidden layer to hidden layer,
        # and so on.
        self.inputs = np.array(inputs).reshape(-1, 1)
        self.hidden_outputs = []
        layer_output = self.inputs

        for i in range(len(self.weights) - 1):
            layer_output = self.relu(np.dot(self.weights[i], layer_output) + self.biases[i])
            self.hidden_outputs.append(layer_output)

        self.output = self.sigmoid(np.dot(self.weights[-1], layer_output) + self.biases[-1])
        return self.output

    def predict(self, inputs):
        output = self.forward(inputs)
        return np.argmax(output)

    # Crossbreeds 2 neural networks to create a child neural network
    def crossover(self, other):
        child = NeuralNetwork(input_size=len(self.weights[0][0]), hidden_layers=[layer.shape[0] for layer in self.weights[:-1]], output_size=self.weights[-1].shape[0])
        for i in range(len(self.weights)):
            child.weights[i] = np.where(np.random.rand(*self.weights[i].shape) > 0.5, self.weights[i], other.weights[i])
            child.biases[i] = np.where(np.random.rand(*self.biases[i].shape) > 0.5, self.biases[i], other.biases[i])
        return child

    # Mutates the neural network biases/weights under a certain condition
    def mutate(self, mutation_rate):
        for i in range(len(self.weights)):
            mutation_mask_w = np.random.rand(*self.weights[i].shape) < mutation_rate
            mutation_mask_b = np.random.rand(*self.biases[i].shape) < mutation_rate
            self.weights[i] += mutation_mask_w * np.random.randn(*self.weights[i].shape)
            self.biases[i] += mutation_mask_b * np.random.randn(*self.biases[i].shape)

# Gives the relevant game state information to the neural network
# Which then allows it to make predictions
# Modifying as needed

def get_game_state(snake_position, fruit_position, snake_body, direction):
    # Relative position of the fruit to the snake
    relative_fruit_x = (fruit_position[0] - snake_position[0]) / window_x
    relative_fruit_y = (fruit_position[1] - snake_position[1]) / window_y

    # Determine relative direction of the fruit
    fruit_direction_up = 1 if relative_fruit_y < 0 else 0
    fruit_direction_down = 1 if relative_fruit_y > 0 else 0
    fruit_direction_left = 1 if relative_fruit_x < 0 else 0
    fruit_direction_right = 1 if relative_fruit_x > 0 else 0

    # Facing direction (encoded as a one-hot vector)
    direction_up = 1 if direction == 'UP' else 0
    direction_down = 1 if direction == 'DOWN' else 0
    direction_left = 1 if direction == 'LEFT' else 0
    direction_right = 1 if direction == 'RIGHT' else 0

    # Danger detection
    danger_left = 1 if (snake_position[0] - 10, snake_position[1]) in snake_body or snake_position[0] - 10 < 0 else 0
    danger_right = 1 if (snake_position[0] + 10, snake_position[1]) in snake_body or snake_position[0] + 10 >= window_x else 0
    danger_up = 1 if (snake_position[0], snake_position[1] - 10) in snake_body or snake_position[1] - 10 < 0 else 0
    danger_down = 1 if (snake_position[0], snake_position[1] + 10) in snake_body or snake_position[1] + 10 >= window_y else 0

    return [
        relative_fruit_x, relative_fruit_y,
        danger_left, danger_right, danger_up, danger_down,
        direction_up, direction_down, direction_left, direction_right,
        fruit_direction_up, fruit_direction_down, fruit_direction_left, fruit_direction_right
    ]




def initialize_population(population_size):
    return [NeuralNetwork() for _ in range(population_size)]


# Fitness function to determine worth of a neural network
# Utilizes steps and distance from fruit to update score
def evaluate_fitness(nn, max_steps=10000, no_progress_steps=1000):
    snake_position = [100, 50]
    snake_body = [[100, 50], [90, 50], [80, 50], [70, 50]]
    fruit_position = [random.randrange(1, (window_x // 10)) * 10, random.randrange(1, (window_y // 10)) * 10]
    fruit_spawn = True
    direction = 'RIGHT'
    steps = 0
    score = 0
    last_distance = np.inf
    progress = False
    direction_change_count = 0

    move_history = []

    # Game loop, if it hits max steps, end the run
    while steps < max_steps:
        # Gets the game state from the get function
        game_state = get_game_state(snake_position, fruit_position, snake_body, direction)
        # Gives the move that the neural network has decided to do based off game state and weights/biases
        prediction = nn.predict(game_state)
        change_to = ['UP', 'DOWN', 'LEFT', 'RIGHT'][prediction]


        # a counter variable, not necessary
        if change_to != direction:
            direction_change_count += 1

        # Changes the direction if needed based off prediction
        if change_to == 'UP' and direction != 'DOWN':
            direction = 'UP'
        if change_to == 'DOWN' and direction != 'UP':
            direction = 'DOWN'
        if change_to == 'LEFT' and direction != 'RIGHT':
            direction = 'LEFT'
        if change_to == 'RIGHT' and direction != 'LEFT':
            direction = 'RIGHT'

        #Changes position of snake based off direcction
        if direction == 'UP':
            snake_position[1] -= 10
        if direction == 'DOWN':
            snake_position[1] += 10
        if direction == 'LEFT':
            snake_position[0] -= 10
        if direction == 'RIGHT':
            snake_position[0] += 10

        snake_body.insert(0, list(snake_position))

        # + 10 to the snakes score if it eats a frit
        if snake_position[0] == fruit_position[0] and snake_position[1] == fruit_position[1]:
            score += 10  # High reward for eating fruit
            fruit_spawn = False
            progress = True
        else:
            snake_body.pop()

        # Spawns a fruit if its eaten
        if not fruit_spawn:
            fruit_position = [random.randrange(1, (window_x // 10)) * 10, random.randrange(1, (window_y // 10)) * 10]
        fruit_spawn = True


        # Ends the run if the snake hits its body or wall, lowers the score
        if (snake_position[0] < 0 or snake_position[0] >= window_x or
                snake_position[1] < 0 or snake_position[1] >= window_y):
            score -= 10
            break

        for block in snake_body[1:]:
            if snake_position[0] == block[0] and snake_position[1] == block[1]:
                score -= 10
                break

        # If the snake is getting closer to the food, it increases the score
        # Motivates over time for the snake to get closer to the food
        # Might be an unnecessary function

        distance_to_food = np.sqrt((snake_position[0] - fruit_position[0]) ** 2 +
                                   (snake_position[1] - fruit_position[1]) ** 2)
        if distance_to_food < last_distance:
            score += 1  # Increased reward for getting closer to food
            progress = True
        else:
            score -= 1  # Penalty for moving away from food

        last_distance = distance_to_food
        steps += 1

        # Functionality to punish snakes that just go in a circle without doing anything
        move_history.append(tuple(snake_position))
        if len(move_history) > no_progress_steps:
            move_history.pop(0)
            if len(set(move_history)) < no_progress_steps // 2:  # Check for loops
                score -= 10  # Penalty for looping
                break

        if steps % no_progress_steps == 0:
            if not progress:
                score -= 10  # Increased penalty for no progress
                break
            progress = False

      #  if direction_change_count < 3 and steps % 50 == 0:  # Penalize for lack of direction change
          #  score -= 200

    return score + steps/25


# Genetic Algorithm, will heavily modify

def genetic_algorithm(population, num_generations, initial_mutation_rate):
    mutation_rate = initial_mutation_rate
    for generation in range(num_generations):
        print(f'Generation {generation + 1}')

        # Evaluates each neural network by running it, and then sorting it based off its' score
        fitness_scores = [evaluate_fitness(nn) for nn in population]
        max_fitness = max(fitness_scores)
        min_fitness = min(fitness_scores)
        scaled_fitness = [(score - min_fitness) / (max_fitness - min_fitness + 1e-8) for score in fitness_scores]

        # Sort the population by scaled fitness scores
        sorted_population = [x for _, x in sorted(zip(scaled_fitness, population), key=lambda item: item[0], reverse=True)]
        new_population = sorted_population[:5]  # Keep the top 5 neural networks intact

        # Generates the new population using the genetic algorithm
        top_count = 5
        while len(new_population) < population_size:
            for i in range(top_count, min(population_size, top_count + 5)):
                parent1 = sorted_population[random.randint(0, top_count - 1)]
                parent2 = sorted_population[random.randint(0, i - 1)]
                child = parent1.crossover(parent2)
                child.mutate(mutation_rate)
                new_population.append(child)
                if len(new_population) >= population_size:
                    break
            top_count += 5

        population = new_population

        # Adjust mutation rate dynamically based on performance
        if max(scaled_fitness) < 0.5:
            mutation_rate = min(1.0, mutation_rate + 0.05)  # Increase mutation rate if performance is low
        else:
            mutation_rate = max(0.01, mutation_rate - 0.01)  # Decrease mutation rate if performance is good

        # Visualization of the top neural networks in the current generation
        for index, nn in enumerate(sorted_population[:visualize_count]):
            visualize_snake(nn, generation + 1, index + 1)
            pygame.display.quit()

    return population

# Visualization function
def visualize_snake(nn, generation, nn_index, max_steps=5000):
    pygame.init()
    game_window = pygame.display.set_mode((window_x, window_y))
    pygame.display.set_caption(f'Snake AI Generation {generation}, NN {nn_index}')

    snake_position = [100, 50]
    snake_body = [[100, 50], [90, 50], [80, 50], [70, 50]]
    fruit_position = [random.randrange(1, (window_x // 10)) * 10, random.randrange(1, (window_y // 10)) * 10]
    fruit_spawn = True
    direction = 'RIGHT'
    steps = 0
    fruits_eaten = 0

    while steps<max_steps:
        game_window.fill(black)

        font = pygame.font.SysFont('arial', 24)
        generation_text = font.render(f'Generation: {generation}, NN: {nn_index} , Fruits Eaten: {fruits_eaten}' ,
                                      True, white)
        game_window.blit(generation_text, [10, 10])

        game_state = get_game_state(snake_position, fruit_position, snake_body, direction)
        prediction = nn.predict(game_state)


        if prediction == 0:
            change_to = 'UP'
        elif prediction == 1:
            change_to = 'DOWN'
        elif prediction == 2:
            change_to = 'LEFT'
        else:
            change_to = 'RIGHT'

        # Prevent snake from reversing direction
        if change_to == 'UP' and direction != 'DOWN':
            direction = 'UP'
        if change_to == 'DOWN' and direction != 'UP':
            direction = 'DOWN'
        if change_to == 'LEFT' and direction != 'RIGHT':
            direction = 'LEFT'
        if change_to == 'RIGHT' and direction != 'LEFT':
            direction = 'RIGHT'

        # Move the snake in the chosen direction
        if direction == 'UP':
            snake_position[1] -= 10
        if direction == 'DOWN':
            snake_position[1] += 10
        if direction == 'LEFT':
            snake_position[0] -= 10
        if direction == 'RIGHT':
            snake_position[0] += 10

        snake_body.insert(0, list(snake_position))

        # Check if the snake has eaten the fruit
        if snake_position[0] == fruit_position[0] and snake_position[1] == fruit_position[1]:
            fruit_spawn = False
            fruits_eaten+=1
        else:
            snake_body.pop()

        # Spawn a new fruit if one has been eaten
        if not fruit_spawn:
            fruit_position = [random.randrange(1, (window_x // 10)) * 10, random.randrange(1, (window_y // 10)) * 10]
        fruit_spawn = True

        # Draw the snake and fruit
        for pos in snake_body:
            pygame.draw.rect(game_window, green, pygame.Rect(pos[0], pos[1], 10, 10))
        pygame.draw.rect(game_window, red, pygame.Rect(fruit_position[0], fruit_position[1], 10, 10))

        # Check for collisions
        if (snake_position[0] < 0 or snake_position[0] >= window_x or
                snake_position[1] < 0 or snake_position[1] >= window_y or
                snake_body[1:].count(snake_position) > 0):
            if (generation+1) % 10==0 and nn_index == 1: # every 10 generations and the best ranked neural network
            # will output to the text file
                with open("edittingtextfile.txt", "a") as file:
                    file.write("Generation "+ str(generation+1)+ ", Fruits Eaten: "+str(fruits_eaten) + "\n")
            break

        pygame.display.update()
        fps.tick(snake_speed)

        steps += 1  # Increment step counter

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return

        # Quit Pygame after visualization to free up memory
    pygame.display.quit()

def main():
    population = initialize_population(population_size)
    genetic_algorithm(population, num_generations, initial_mutation_rate)
    pygame.quit()


if __name__ == "__main__":
    main()
