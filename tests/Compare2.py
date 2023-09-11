import tensorflow as tf
import numpy as np
import argparse

import os

from collections import deque
import timeit
import random

from environment.RiverCrossingEnv import RiverCrossingEnv
from agent import AgentModel as ag

MIN_REPLAY_SIZE = 1000

RANDOM_SEED = 5
tf.random.set_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# minimum rewards allowed while training
# this is done to avoid being stuck
MIN_TRAIN_REWARDS = -1000

def log(txt,agent_type,  bellman_update, type, lamb):
    with open('./compare2/logs/result_{}_{}_{}_{}.log'.format(agent_type, bellman_update, type, lamb), 'a') as f:
        f.write(txt + '\n')
    print(txt)


def run():
    parser = argparse.ArgumentParser(description='Run QL for River Crossing domain.')
    parser.add_argument('-b', '--bellman_update', default='Target', help='The type of Bellman update Target, TD or LSE, default Target.')
    parser.add_argument('-l', '--lamb', type=float, default=0.0, help='The risk param, default to 0.0.')
    parser.add_argument('-t', '--type', default='DQN_CACHED',
                        help='The type of algorithm QL, DQN, DQN_CONV, DQN_CACHED, DQN_CONV_CACHED, DQN_SKIP, DQN_CONV_SKIP, default QL.')
    args = parser.parse_args()

    if not os.path.exists('compare2/logs'):
        os.makedirs('compare2/logs')

    if not os.path.exists('compare2/models'):
        os.makedirs('compare2/models')

    bellman_update = args.bellman_update
    lamb = args.lamb
    alpha = 0.1
    gamma = 0.99

    epsilon = 0.3  # Epsilon-greedy algorithm in initialized at 1 meaning every step is random at the start
    epsilon_final = 0.1

    # An episode a full game
    train_episodes = 150

    shape = (10, 10)

    # Building environment
    env = RiverCrossingEnv(shape, state_as_img=True)
    agent_type = args.type

    h, w = env.shape

    env.state_as_img = agent_type.startswith('DQN_CONV')

    policy_stack = {}

    for y in range(h):
        for x in range(w):
            s = x + (y * w)
            policy_stack[s] = []

    samples = 20
    for sample in range(1, samples+1):

        # 1. Initialize the Target and Main models
        # Main Model (updated every step)
        model = ag.AgentModel.build(agent_type, bellman_update, env, alpha, gamma, lamb)
        model.load_model()

        # Target Model (updated every 100 steps)
        target_model = ag.AgentModel.build(agent_type, bellman_update, env, alpha, gamma, lamb)
        target_model.set_weights(model)

        replay_memory = deque(maxlen=1_000)

        steps_to_update_target_model = 0

        epsilon_decay = 1.0

        for episode in range(1, train_episodes+1):
            total_training_rewards = 0
            state = env.reset()
            done = False
            while not done:
                steps_to_update_target_model += 1

                # 2. Explore using the Epsilon Greedy Exploration Strategy
                random_number = np.random.rand()
                if random_number <= epsilon_decay:
                    # Explore
                    action = env.action_space.sample()
                else:
                    # Exploit best known action
                    predicted = model.find_qs(state)
                    action = np.argmax(predicted)

                # Epsilon decay to make learning faster
                epsilon_decay -= 0.1
                if epsilon_decay < epsilon:
                    epsilon_decay = epsilon

                if episode >= 75:
                    epsilon_decay = epsilon_final


                # step
                if total_training_rewards < MIN_TRAIN_REWARDS:
                    new_state, reward, done, info = env.step_safe()
                else:
                    new_state, reward, done, info = env.step(action)

                # keeping replay memory to batch training
                replay_memory.append([state, action, reward, new_state, done])

                # 3. Update the Main Network using the Bellman Equation
                if steps_to_update_target_model % 4 == 0 or done:
                    train(replay_memory, model, target_model)

                state = new_state
                total_training_rewards += reward

                if done:
                    safe_points = model.find_safe_points()
                    log('{},{},{},{},{}'.format(lamb, sample, episode, total_training_rewards, safe_points), agent_type, bellman_update, 'train', lamb)

                    target_model.set_weights(model)
                    steps_to_update_target_model = 0

                    break

        target_model.save_model(
            'compare2/models/model_{}_{}_{}_{}.h5'.format(agent_type, bellman_update, lamb, sample))


def run_model(env, policy, lamb, agent_type, bellman_update):
    samples = 1000
    for sample in range(1, samples + 1):
        total_rewards = 0
        state = env.reset()
        done = False
        while not done:
            # Exploit best known action
            action = policy[state]

            # step
            new_state, reward, done, info = env.step(action)

            state = new_state
            total_rewards += reward

            if done:
                log('{},{},{}'.format(lamb, sample, total_rewards), agent_type, bellman_update, 'run', lamb)
                break
def train(replay_memory, model, target_model):

    if len(replay_memory) < MIN_REPLAY_SIZE:
        return

    batch_size = 100
    mini_batch = random.sample(replay_memory, batch_size)

    model.update_model(mini_batch, target_model)


def main():
    try:
        # record start time
        t_0 = timeit.default_timer()

        # running
        run()

        # calculate elapsed time and print
        t_1 = timeit.default_timer()
        elapsed_time = round((t_1 - t_0), 3)
        print(f"Elapsed time: {elapsed_time} s")
    except (ValueError, OverflowError) as error:
        print(error)
        print('Error')


if __name__ == "__main__":
    main()



