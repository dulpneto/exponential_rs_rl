import tensorflow as tf
import numpy as np
import sys
import argparse

import os

from collections import deque
import timeit
import random
import math

from environment.RiverCrossingEnv import RiverCrossingEnv
from agent import AgentModel as ag
from agent.utils import UnderflowError

MIN_REPLAY_SIZE = 1000

RANDOM_SEED = 5
tf.random.set_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# minimum rewards allowed while training
# this is done to avoid being stuck
MIN_TRAIN_REWARDS = -5000

UNDERFLOW_THRESHOLD = 1000

RUN_FOR = 'ql2'
#RUN_FOR = 'dqn2'

def log(txt, type, bellman_update, alpha):
    fpath = './logs/{}/result_env_{}_{}.log'.format(RUN_FOR, alpha, type)

    with open(fpath, 'a') as f:
        f.write(txt + '\n')
    print(txt)



def run(bellman_update, env, alpha):

    if not os.path.exists('logs/{}'.format(RUN_FOR)):
        os.makedirs('logs/{}'.format(RUN_FOR))

    gamma = 0.99

    epsilon = 0.3  # Epsilon-greedy algorithm in initialized at 1 meaning every step is random at the start

    # An episode a full game
    train_episodes = 300

    # Building environment
    #env = RiverCrossingEnv(shape, state_as_img=False)
    if RUN_FOR == 'ql2':
        agent_type = 'QL'
    else:
        agent_type = 'DQN_CACHED'

    h, w = env.shape

    #for lamb in [-1.5, -1.0, -0.75, -0.5, -0.2, -0.1, 0.1, 0.2, 0.5, 0.75, 1.0, 1.5]:
    #for lamb in [-0.2,0.75,1.0]:
    #for lamb in [-1.5, -0.5, 0.5, 1.5]:

    samples = 15
    for sample in range(1, samples + 1):

        for lamb in [-1.5, -0.5, 0.5, 1.5]:

            if lamb == 0:
                print('skip 0')
                continue

            # 1. Initialize the Target and Main models
            # Main Model (updated every step)
            model = ag.AgentModel.build(agent_type, bellman_update, env, alpha, gamma, lamb)
            model.load_model()

            # Target Model (updated every 100 steps)
            target_model = ag.AgentModel.build(agent_type, bellman_update, env, alpha, gamma, lamb)
            target_model.set_weights(model)

            replay_memory = deque(maxlen=1_000)

            steps_to_update_target_model = 0

            policy_stack = {}

            for y in range(h):
                for x in range(w):
                    s = x + (y * w)
                    policy_stack[s] = []

            error=False

            for episode in range(1, train_episodes+1):

                if error:
                    break

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
                    if total_training_rewards < (MIN_TRAIN_REWARDS*env.max_abs_r):
                        new_state, reward, done, info = env.step_safe()
                    else:
                        new_state, reward, done, info = env.step(action)

                    # keeping replay memory to batch training
                    replay_memory.append([state, action, reward, new_state, done])

                    # 3. Update the Main Network using the Bellman Equation
                    if steps_to_update_target_model % 4 == 0 or done:
                        try:
                            train(replay_memory, model, target_model)
                        except (ValueError, OverflowError):
                            log('{}\t{}\t{}\t{}'.format(lamb, bellman_update, episode, 1), 'error', bellman_update, h*w)
                            error = True
                            break
                        except (UnderflowError):
                            log('{}\t{}\t{}\t{}'.format(lamb, bellman_update, episode, 1), 'underflow', bellman_update,h*w)
                            error = True
                            break

                    state = new_state
                    total_training_rewards += reward

                    if done:

                        if episode > train_episodes-50:
                            for y in range(h):
                                for x in range(w):
                                    s = x + (y * w)
                                    predicted = model.find_qs(s)
                                    action = np.argmax(predicted)
                                    policy_stack[s].append(action)

                        predicted = model.find_qs(env.s0)
                        vs0 = max(predicted)

                        if math.isnan(vs0):
                            #log('{}\t{}\t{}\t{}\t{}'.format(lamb, bellman_update, sample, episode, 1), 'error', bellman_update, alpha)
                            error = True
                            break

                        log('{}\t{}\t{}\t{}\t{}'.format(lamb, bellman_update, sample, episode, vs0), 'vso', bellman_update, h*w)
                        log('{}\t{}\t{}\t{}\t{}'.format(lamb, bellman_update, sample, episode, model.bellman.max_u), 'us', bellman_update, h*w)
                        log('{}\t{}\t{}\t{}\t{}'.format(lamb, bellman_update, sample, episode, model.bellman.min_u), 'us_min',
                            bellman_update, h*w)
                        log('{}\t{}\t{}\t{}\t{}'.format(lamb, bellman_update, sample, episode, total_training_rewards), 'rewards', bellman_update, h*w)
                        model.bellman.max_u = 0
                        model.bellman.min_u = sys.maxsize

                        safe_points = model.find_safe_points()
                        log('{}\t{}\t{}\t{}\t{}'.format(lamb, bellman_update, sample, episode, safe_points), 'safe', bellman_update, h*w)

                        target_model.set_weights(model)
                        steps_to_update_target_model = 0
                        break

                max_q = max([max(abs(model.Q[s])) for s in range(10)])
                print('{}\t{}\t{}'.format(sample, episode, max_q))
            if len(policy_stack[env.s0]) > 0:
                policy = {}
                for y in range(h):
                    for x in range(w):
                        s = x + (y * w)
                        a = np.argmax(np.bincount(policy_stack[s]))
                        policy[s] = a

                proper_policy = env.check_proper_policy(policy)

                if proper_policy:
                    log('{}\t{}\t{}\t{}'.format(lamb, bellman_update, sample, 1), 'proper',
                        bellman_update, h*w)
                else:
                    if model.bellman.under_count >= UNDERFLOW_THRESHOLD:
                        log('{}\t{}\t{}\t{}'.format(lamb, bellman_update, sample, 1), 'underflow', bellman_update,
                            h*w)
                    else:
                        log('{}\t{}\t{}\t{}'.format(lamb, bellman_update, sample, 1), 'not_proper',
                            bellman_update, h*w)


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





        parser = argparse.ArgumentParser(description='Run QL for River Crossing domain.')
        parser.add_argument('-b', '--bellman_update', default='TD',
                            help='The type of Bellman update Target, TD or SI, default Target.')

        parser.add_argument('-a', '--alpha', type=float, default=0.1, help='The learning rate, default to 0.1.')
        args = parser.parse_args()

        bellman_update = args.bellman_update

        alpha = args.alpha

        # 36
        # 64
        # 144
        # 196

        for s in [6, 8, 12, 14]:
            shape = (s, s)
            # Building environment
            env = RiverCrossingEnv(shape, state_as_img=False)

            # running
            if RUN_FOR == 'ql2':
                run('Target', env, alpha)
                #run('LSE', env, alpha)

                run('TD', env, alpha)
                run('TD_TRUNC', env, alpha)
                run('SI', env, alpha)
            else:
                run(bellman_update, env, alpha)

        # calculate elapsed time and print
        t_1 = timeit.default_timer()
        elapsed_time = round((t_1 - t_0), 3)
        print(f"Elapsed time: {elapsed_time} s")
    except (ValueError, OverflowError) as error:
        print(error)
        print('Error')


if __name__ == "__main__":
    main()



