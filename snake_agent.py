import numpy as np
import pygame
from collections import deque

import keras
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten

import snakpe


def render_game(surface, game):
    surface.fill((0, 0, 0))
    for x in range(game_x):
        for y in range(game_y):
            if x == game.apple[0] and y == game.apple[1]:
                pygame.draw.rect(surface,
                                 (255, 0, 0),
                                 (x * cell_size, y * cell_size, cell_size, cell_size),
                                 0)
            elif game.grid[x][y] == 1:
                pygame.draw.rect(surface,
                                 (255, 255, 255),
                                 (x * cell_size, y * cell_size, cell_size, cell_size),
                                 0)
    pygame.display.update()
    pygame.time.delay(100)


def construct_model():
    nb_actions = env.action_space_n
    model = Sequential()
    model.add(Dense(256, input_dim=env.obs_space_n, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    # model.add(Dense(32, input_dim=env.obs_space_n, activation='relu'))
    # model.add(Dense(32, activation='relu'))
    # model.add(Dense(32, activation='relu'))
    model.add(Dense(nb_actions, activation='linear'))
    model.compile(loss="mse", optimizer="adam")
    print(model.summary())
    return model


def run_episode(max_turns, memory, epsilon, model, screen=None):
    env.reset()
    total_score = 0
    for j in range(max_turns):

        if screen is not None:
            render_game(screen, env)

        state, done = env.get_game_state()
        if done:
            break

        if np.random.uniform(0, 1) < epsilon:
            action = np.random.randint(5)
        else:
            action = np.argmax(model.predict(state))

        if action != 4 and (env.direction - action) % 2 != 0:
            env.direction = action

        result = env.move()
        new_state, _ = env.get_game_state()
        if not isinstance(result, bool):
            score = 0
        elif result:
            score = 1 #env.length
        elif not result:
            score = -1
        else:
            raise ValueError("Invalid result returned from snake")

        total_score += score
        if memory is not None:
            memory.append((state, new_state, score, action, done))

    return total_score


def replay_memory(model, memory, batch_size=512, epochs=3):
    num = len(memory)
    if num >= batch_size:
        idx = np.random.randint(num-1, size=batch_size)
        x = []
        y = []
        for i in idx:
            state, new_state, score, action, done = memory[i]
            if done:
                rewards = [score for _ in range(env.action_space_n)]
            else:
                rewards = model.predict(state)[0]
                rewards[action] = score + gamma * np.max(model.predict(new_state))
            x.append(state[0])
            y.append(rewards)
        model.fit(np.array(x), np.array(y), batch_size=batch_size, epochs=epochs, verbose=0)


def train_deep_qlearn(model, visualise=False, epsilon=1.0, num_episodes=5000, memory_limit=50000):
    memory = deque(maxlen=memory_limit)

    if visualise:
        screen_width = game_x * cell_size
        screen_height = game_y * cell_size
        screen = pygame.display.set_mode((screen_width, screen_height))

    cumulative_total = 0
    n_ep = 0
    for i in range(num_episodes):
        if epsilon > eps_min:
            epsilon -= epsilon_decay
        if visualise:
            cumulative_total += run_episode(episode_max_turns, memory, epsilon, model, screen)
        else:
            cumulative_total += run_episode(episode_max_turns, memory, epsilon, model)
        n_ep += 1
        if i % 50 == 0:
            print("Episode {} average score: {}".format(i, cumulative_total/n_ep))
            cumulative_total = 0
            n_ep = 0
        replay_memory(model, memory)

    print("Training finished")
    model.save('snake.h5')

    if visualise:
        for i in range(5):
            result = run_episode(10000000000, None, 0, model, screen)
            print("Testing episode {} result: {}".format(i, result))
            pygame.time.delay(500)


def visualise_episodes(model, num_episodes=5):
    screen_width = game_x * cell_size
    screen_height = game_y * cell_size
    screen = pygame.display.set_mode((screen_width, screen_height))

    for i in range(num_episodes):
        result = run_episode(10000000000, None, 0, model, screen)
        print("Running episode {} result: {}".format(i, result))
        pygame.time.delay(1000)


if __name__ == "__main__":

    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Restrict TensorFlow to only use the fourth GPU
            tf.config.experimental.set_visible_devices(gpus[0], 'GPU')

            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)

    np.random.seed(123)
    game_x = 10
    game_y = 10
    cell_size = 30
    env = snakpe.Snake(game_x, game_y)

    eps_min = 0.1
    epsilon_decay = 0
    gamma = 0.95
    episode_max_turns = 100000

    #model = construct_model()
    model = keras.models.load_model('snake head.h5')
    visualise_episodes(model)
    #train_deep_qlearn(model, visualise=False, num_episodes=5000, epsilon=0.4)
