import tensorflow as tf
import numpy as np

import argparse

import timeit

from agent import BellmanUpdate as bu

from environment.RiverCrossingEnv import RiverCrossingEnv
from environment.TwoArmedBanditEnv import TwoArmedBandit
from planning.ValueIteration import ValueIteration as ValueIteration

MIN_REPLAY_SIZE = 1000

RANDOM_SEED = 5
tf.random.set_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

QUIET = False


def run():

    parser = argparse.ArgumentParser(description='Run DQN for River Crossing domain.')
    parser.add_argument('-env', '--env_type', default='RIVER',
                        help='Domain RIVER or TWO_ARMED, default RIVER')
    parser.add_argument('-b', '--bellman_update', default='Target', help='The type of Bellman update Target, TD or LSE, default Target.')
    parser.add_argument('-l', '--lamb', type=float, default=0.0, help='The risk param, default to 0.0.')
    parser.add_argument('-g', '--gamma', type=float, default=0.99, help='The discount factor, default to 0.99.')
    parser.add_argument('-a', '--alpha', type=float, default=0.1, help='The learning rate, default to 0.1.')

    # River exclusive
    parser.add_argument('-sh', '--shape_h', type=int, default=10, help='The shape h size, default to 5.')
    parser.add_argument('-sw', '--shape_w', type=int, default=10, help='he shape w size, default to 4.')

    # Two-Armed exclusive
    parser.add_argument('-a0', '--arm_0_r', type=float, default=0.0, help='The Arm 0 reward, default to 0.')
    parser.add_argument('-a1', '--arm_1_mean', type=float, default=0.0, help='The Arm 1 mean reward, default to 0.')

    args = parser.parse_args()

    env_type = args.env_type

    bellman_update = args.bellman_update
    alpha = args.alpha
    gamma = args.gamma
    lamb = args.lamb


    # Building environment
    if env_type == 'RIVER':
        shape = (args.shape_h, args.shape_w)
        env = RiverCrossingEnv(shape)
    elif env_type == 'TWO_ARMED':
        env = TwoArmedBandit(args.arm_0_r, args.arm_1_mean)

    if bellman_update == bu.Type.TARGET:
        policy, V, steps, updates, diffs, V_history, utilities, utilities_min = ValueIteration.run_target(env, lamb, gamma, alpha)
    elif bellman_update == bu.Type.TD:
        policy, V, steps, updates, diffs, V_history, utilities, utilities_min = ValueIteration.run_td(env, lamb, gamma, alpha)
    elif bellman_update == bu.Type.TD_TRUNC:
        policy, V, steps, updates, diffs, V_history, utilities, utilities_min = ValueIteration.run_td(env, lamb, gamma, alpha, trunc=True)
    elif bellman_update == bu.Type.SOFT_INDICATOR:
        policy, V, steps, updates, diffs, V_history, utilities, utilities_min = ValueIteration.run_soft_indicator(env, lamb, gamma, alpha)
    else:
        raise Exception("Not implemented {}".format(bellman_update))

    safe_points = ValueIteration.find_safe_points(env, policy)

    print('Policy', policy)
    print('V', V)
    print('steps', steps)
    print('updates', updates)
    print('safe_points', safe_points)


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



