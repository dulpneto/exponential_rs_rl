import tensorflow as tf
import numpy as np

import os

from planning.ValueIteration import ValueIteration as ValueIteration

from agent import BellmanUpdate as bu

from collections import defaultdict

import statistics as st

import timeit
import math

from environment.TwoArmedBanditEnv import TwoArmedBandit
from agent.utils import UnderflowError

MIN_REPLAY_SIZE = 1000

RANDOM_SEED = 5
tf.random.set_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# minimum rewards allowed while training
# this is done to avoid being stuck
MIN_TRAIN_REWARDS = -5000

RUN_FOR = 'ql2'
#RUN_FOR = 'dqn2'

def log(txt, type, bellman_update, lamb):
    fpath = './logs_two/{}/4-result_lamb_{}_{}.log'.format(RUN_FOR, lamb, type)

    with open(fpath, 'a') as f:
        f.write(txt + '\n')
    print(txt)



def run(bellman_update, lamb):

    if not os.path.exists('logs_two/{}'.format(RUN_FOR)):
        os.makedirs('logs_two/{}'.format(RUN_FOR))

    gamma = 0.99

    epsilon = 0.3  # Epsilon-greedy algorithm in initialized at 1 meaning every step is random at the start

    # An episode a full game
    train_episodes = 150

    #for l in range(-15, 16, 1):
    #    lamb = l / 10

    for alpha in [0.2, 0.1, 0.05, 0.01]:

        bellman = bu.BellmanUpdate.build(bellman_update, alpha, gamma, lamb)

        samples = 100
        for sample in range(1, samples+1):

            for i in range(0, 31):
                arm_0_r = (-0.5 + (i * (1 / 30)))
                for j in range(0, 31):
                    arm_1_mean = (-0.5 + (j * (1 / 30)))

                    env = TwoArmedBandit(arm_0_r, arm_1_mean)
                    #if not (arm_0_r == 0.5 and arm_1_mean == 0):
                    #    continue

                    #if sample == 1:
                    #    policy, V, steps, updates, diffs, V_history, utilities = ValueIteration.run_target(env, lamb, gamma, alpha)
                    #    print('V', V)

                    # 1. Initialize the Target and Main models
                    # Main Model (updated every step)

                    error=False

                    calcs = []

                    Q = defaultdict(lambda: np.ones(env.action_space.n) * bellman.default_V)

                    for episode in range(1, train_episodes+1):

                        if error:
                            break

                        try:
                            env.reset()
                            current_state = env.current_s

                            # 2. Explore using the Epsilon Greedy Exploration Strategy
                            random_number = np.random.rand()
                            if random_number <= epsilon:
                                # Explore
                                action = env.action_space.sample()
                            else:
                                # Exploit best known action
                                predicted = Q[current_state]
                                action = np.argmax(predicted)


                            new_state, reward, done, info = env.step(action)

                            future_qs = Q[new_state]
                            current_qs = Q[current_state]
                            Q[current_state][action] = bellman.bellman_update(current_qs, future_qs, action, reward)

                        except OverflowError:
                            log('{}\t{}\t{}\t{}'.format(alpha, bellman_update, episode, 1), 'error', bellman_update, lamb)
                            error = True
                            break
                        except UnderflowError:
                            log('{}\t{}\t{}\t{}'.format(alpha, bellman_update, episode, 1), 'error-under', bellman_update, lamb)
                            error = True
                            break
                        except ValueError:
                            log('{}\t{}\t{}\t{}'.format(alpha, bellman_update, episode, 1), 'error-value', bellman_update, lamb)
                            error = True
                            break


                        predicted = Q[env.s0]
                        vs0 = max(predicted)

                        if math.isnan(vs0):
                            log('{}\t{}\t{}\t{}'.format(alpha, bellman_update, episode, 1), 'error-isnan', bellman_update, lamb)
                            error = True
                            break

                        calc = 1 - (np.argmax(predicted)*2)
                        if episode >= (train_episodes-10):
                            calcs.append(calc)

                        if episode == train_episodes:
                            log('{}\t{}\t{}\t{}\t{}\t{}'.format(alpha, bellman_update, episode, i, j, st.mode(calcs)), 'calc', bellman_update,
                                lamb)

                            #print(Q[0], Q[1], calc, st.mode(calcs))
                            #if np.argmax(predicted) > 0:
                            #    print("AQUI")
                        #log('{}\t{}\t{}\t{}'.format(lamb, bellman_update, episode, vs0), 'vso', bellman_update, alpha)
                        #log('{}\t{}\t{}\t{}'.format(lamb, bellman_update, episode, model.bellman.max_u), 'us', bellman_update, alpha)
                        #log('{}\t{}\t{}\t{}'.format(lamb, bellman_update, episode, total_training_rewards), 'rewards', bellman_update, alpha)


def main():
    try:
        # record start time
        t_0 = timeit.default_timer()


        lamb = 5.0

        # running
        run('Target', lamb)
        #run('TD', lamb)
        #run('SI', lamb)

        # calculate elapsed time and print
        t_1 = timeit.default_timer()
        elapsed_time = round((t_1 - t_0), 3)
        print(f"Elapsed time: {elapsed_time} s")
    except (ValueError, OverflowError) as error:
        print(error)
        print('Error')


if __name__ == "__main__":
    main()



