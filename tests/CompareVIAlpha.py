import tensorflow as tf
import numpy as np
import os

import argparse

from collections import deque
import timeit
import random

from environment.RiverCrossingEnv import RiverCrossingEnv
from agent import AgentModel as ag
from planning.ValueIteration import ValueIteration as ValueIteration

MIN_REPLAY_SIZE = 1000

RANDOM_SEED = 5
tf.random.set_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# minimum rewards allowed while training
# this is done to avoid being stuck
MIN_TRAIN_REWARDS = -5000

def log(txt, type):
    with open('./logs/vi/alpha_lamb1.5_result_{}.log'.format(type), 'a') as f:
        f.write(txt + '\n')
    print(txt)

def log_list(txt, type, list):
    i = 0
    for d in list:
        i += 1
        log(txt.format(i, d), type)

def log_list_2(txt, type, list, k):
    i = 0
    for d in list:
        i += 1
        log(txt.format(i, d[k]), type)

def run_vi():

    if not os.path.exists('logs/vi'):
        os.makedirs('logs/vi')

    #alpha = 0.1
    gamma = 0.99

    shape = (10, 10)

    # Building environment
    env = RiverCrossingEnv(shape, state_as_img=False)

    lamb = 1.5

    policy, V, steps, updates, diffs, V_history, utilities = ValueIteration.run_target(env, lamb, gamma, 1.0)

    for a in range(10, -1, -1):
        alpha = a / 10

        if alpha == 0.0:
            alpha = 0.01

        print('\nAlpha {}'.format(alpha))
        policy_2, V_2, steps_2, updates_2, diffs_2, V_history_2, utilities_2 = ValueIteration.run_target(env, lamb, gamma, alpha)

        if alpha <= 0.8 or lamb < 1.5:
            policy_shen, V_shen, steps_shen, updates_shen, diffs_shen, V_history_shen, utilities_shen = ValueIteration.run_td(env, lamb,
                                                                                                            gamma, alpha)

            print('\tVI\tTarget\tTD')
            log('{}\t{}\t{}\t{}'.format(alpha,steps, steps_2, steps_shen), 'steps')
            log_list('{}'.format(alpha) + '\tTD\t{}\t{}', 'diffs', diffs_shen)
            log_list_2('{}'.format(alpha) + '\tTD\t{}\t{}', 'vso', V_history_shen, env.s0)
            log_list('{}'.format(alpha) + '\tTD\t{}\t{}', 'us', utilities_shen)

        log_list('{}'.format(alpha)+'\tVI\t{}\t{}', 'diffs', diffs)
        log_list('{}'.format(alpha) + '\tTarget\t{}\t{}', 'diffs', diffs_2)


        log_list_2('{}'.format(alpha) + '\tVI\t{}\t{}', 'vso', V_history, env.s0)
        log_list_2('{}'.format(alpha) + '\tTarget\t{}\t{}', 'vso', V_history_2, env.s0)


        log_list('{}'.format(alpha) + '\tTarget\t{}\t{}', 'us', utilities_2)


def run_ql():

    if not os.path.exists('logs/vi'):
        os.makedirs('logs/vi')

    alpha = 0.1
    gamma = 0.99

    epsilon = 0.3

    shape = (10, 10)

    train_episodes = 150

    # Building environment
    env = RiverCrossingEnv(shape, state_as_img=False)



    lambs = []
    for l in range(-15, 16, 1):
        lamb = l / 10

        if lamb == 0:
            print('skip 0')
            continue

        lambs.append(lamb)

        model = ag.AgentModel.build('QL', 'Target', env, alpha, gamma, lamb)
        model.load_model()

        for episode in range(1, train_episodes + 1):
            total_training_rewards = 0
            state = env.reset()
            done = False
            while not done:
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
                replay_memory = []
                replay_memory.append([state, action, reward, new_state, done])

                model.update_model(replay_memory, model)

                state = new_state
                total_training_rewards += reward

                if done:
                    safe_points = model.find_safe_points()
                    break


        policy, V, steps, updates, diffs, V_history = ValueIteration.run(env, lamb, gamma)
        policy_2, V_2, steps_2, updates_2, diffs_2, V_history_2, utilities_2 = ValueIteration.run_target(env, lamb, gamma, alpha)
        policy_shen, V_shen, steps_shen, updates_shen, diffs_shen, V_history_shen, utilities_shen = ValueIteration.run_td(env, lamb,
                                                                                                            gamma,
                                                                                                            alpha)
        print('\nRisk {}'.format(lamb))
        print('\tVI\tTarget\tTD')
        log('{}\t{}\t{}\t{}'.format(lamb,steps, steps_2, steps_shen), 'steps')

        log_list('{}'.format(lamb)+'\tVI\t{}\t{}', 'diffs', diffs)
        log_list('{}'.format(lamb) + '\tTarget\t{}\t{}', 'diffs', diffs_2)
        log_list('{}'.format(lamb) + '\tTD\t{}\t{}', 'diffs', diffs_shen)

        log_list_2('{}'.format(lamb) + '\tVI\t{}\t{}', 'vso', V_history, env.s0)
        log_list_2('{}'.format(lamb) + '\tTarget\t{}\t{}', 'vso', V_history_2, env.s0)
        log_list_2('{}'.format(lamb) + '\tTD\t{}\t{}', 'vso', V_history_shen, env.s0)

        log_list('{}'.format(lamb) + '\tTarget\t{}\t{}', 'us', utilities_2)
        log_list('{}'.format(lamb) + '\tTD\t{}\t{}', 'us', utilities_shen)

def main():
    try:
        # record start time
        t_0 = timeit.default_timer()

        # running
        run_vi()

        # calculate elapsed time and print
        t_1 = timeit.default_timer()
        elapsed_time = round((t_1 - t_0), 3)
        print(f"Elapsed time: {elapsed_time} s")
    except (ValueError, OverflowError) as error:
        print(error)
        print('Error')


if __name__ == "__main__":
    main()



