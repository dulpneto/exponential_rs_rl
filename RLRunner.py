import tensorflow as tf
import numpy as np

import argparse

from collections import deque
import timeit
import random

from environment.RiverCrossingEnv import RiverCrossingEnv
from environment.TwoArmedBanditEnv import TwoArmedBandit
from agent import AgentModel as ag

MIN_REPLAY_SIZE = 1000

RANDOM_SEED = 5
tf.random.set_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

QUIET = False


def run():

    parser = argparse.ArgumentParser(description='Run DQN for River Crossing domain.')
    parser.add_argument('-env', '--env_type', default='RIVER',
                        help='Domain RIVER or TWO_ARMED, default RIVER')
    parser.add_argument('-t', '--type', default='QL', help='The type of algorithm QL, DQN, DQN_CONV, DQN_CACHED, DQN_CONV_CACHED, DQN_SKIP, DQN_CONV_SKIP, default QL. CONV methods are not implemented on Two Armed domain.')
    parser.add_argument('-b', '--bellman_update', default='Target', help='The type of Bellman update Target, TD or LSE, default Target.')
    parser.add_argument('-l', '--lamb', type=float, default=0.0, help='The risk param, default to 0.0.')
    parser.add_argument('-g', '--gamma', type=float, default=0.99, help='The discount factor, default to 0.99.')
    parser.add_argument('-a', '--alpha', type=float, default=0.1, help='The learning rate, default to 0.1.')
    parser.add_argument('-p', '--epsilon', type=float, default=0.3, help='The epsilon, default to 0.3.')
    parser.add_argument('-e', '--episodes', type=int, default=300, help='The numer of episodes, default to 150.')

    # River exclusive
    parser.add_argument('-sh', '--shape_h', type=int, default=10, help='The shape h size, default to 5.')
    parser.add_argument('-sw', '--shape_w', type=int, default=10, help='he shape w size, default to 4.')

    # Two-Armed exclusive
    parser.add_argument('-a0', '--arm_0_r', type=float, default=0.0, help='The Arm 0 reward, default to 0.')
    parser.add_argument('-a1', '--arm_1_mean', type=float, default=0.0, help='The Arm 1 mean reward, default to 0.')

    args = parser.parse_args()

    env_type = args.env_type

    agent_type = args.type
    bellman_update = args.bellman_update
    alpha = args.alpha
    gamma = args.gamma
    lamb = args.lamb

    epsilon = args.epsilon  # Epsilon-greedy algorithm in initialized at 1 meaning every step is random at the start

    # An episode a full game
    train_episodes = args.episodes

    # if we are working with convolutional network we must return stats as images
    state_as_img = agent_type.startswith('DQN_CONV')

    # Building environment
    if env_type == 'RIVER':
        shape = (args.shape_h, args.shape_w)
        env = RiverCrossingEnv(shape, state_as_img=state_as_img)
    elif env_type == 'TWO_ARMED':
        env = TwoArmedBandit(args.arm_0_r, args.arm_1_mean)

    # 1. Initialize the Target and Main models
    # Main Model (updated every step)
    model = ag.AgentModel.build(agent_type, bellman_update, env, alpha, gamma, lamb)
    model.load_model()

    # Target Model (updated every 100 steps)
    target_model = ag.AgentModel.build(agent_type, bellman_update, env, alpha, gamma, lamb)
    target_model.set_weights(model)

    replay_memory = deque(maxlen=1_000)

    steps_to_update_target_model = 0

    for episode in range(train_episodes):
        if not QUIET:
            print('Episode: {}'.format(episode))

        total_training_rewards = 0
        state = env.reset()
        done = False
        while not done:
            steps_to_update_target_model += 1

            # 2. Explore using the Epsilon Greedy Exploration Strategy
            random_number = np.random.rand()
            if random_number <= epsilon:
                # Explore
                action = env.action_space.sample()
            else:
                # Exploit best known action
                predicted = model.find_qs(state)
                action = np.argmax(predicted)

            # step
            new_state, reward, done, info = env.step(action)

            # keeping replay memory to batch training
            replay_memory.append([state, action, reward, new_state, done])

            # 3. Update the Main Network using the Bellman Equation
            if steps_to_update_target_model % 4 == 0 or done:
                train(replay_memory, model, target_model)

            state = new_state
            total_training_rewards += reward

            if done:
                if not QUIET:
                    print('Total training rewards: {} after n steps = {} with final reward = {}'.format(
                        total_training_rewards, episode, reward))
                total_training_rewards += 1
                if not QUIET:
                    env.render(model)
                    model.print_qs_model()
                    print('Copying main network weights to the target network weights')
                target_model.set_weights(model)
                steps_to_update_target_model = 0
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
        if not QUIET:
            print(f"Elapsed time: {elapsed_time} s")
    except (ValueError, OverflowError) as error:
        print(error)
        print('Error')


if __name__ == "__main__":
    main()



