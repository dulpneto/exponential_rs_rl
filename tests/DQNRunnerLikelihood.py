import sys

import tensorflow as tf
import numpy as np

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

QUIET = True


def run(env, model, target_model, epsilon, policy_control, threshold_likelihood):

    replay_memory = deque(maxlen=1_000)

    all_likelihood = []

    likelihood_control = find_likelihood(env.shape, policy_control, epsilon)

    steps_to_update_target_model = 0

    episode = 0
    while True:
        episode += 1
        if not QUIET:
            print('Episode: {}'.format(episode))

        if episode > 1_000:
            return 'ERROR_EPS'

        total_training_rewards = 0
        state = env.reset()

        likelihood = 1

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

            # Calculating likelihood
            if policy_control[state] == action:
                likelihood = likelihood * ((1 - epsilon) + (epsilon / 4))
            else:
                likelihood = likelihood * (epsilon / 4)

            # step
            new_state, reward, done, info = env.step(action)

            # keeping replay memory to batch training
            replay_memory.append([state, action, reward, new_state, done])

            # 3. Update the Main Network using the Bellman Equation
            if steps_to_update_target_model % 4 == 0 or done:
                train(replay_memory, model, target_model)

            state = new_state
            total_training_rewards += reward

            if steps_to_update_target_model > 10_000:
                return 'ERROR_EXPLORATION'

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

        all_likelihood.append(likelihood)
        if np.mean(all_likelihood[-10:]) >= (likelihood_control * threshold_likelihood):
            if not QUIET:
                print('Likelihood Reached')
            return episode


def train(replay_memory, model, target_model):

    if len(replay_memory) < MIN_REPLAY_SIZE:
        return

    batch_size = 100
    mini_batch = random.sample(replay_memory, batch_size)

    model.update_model(mini_batch, target_model)


def find_safe_points(shape, policy):

    count = 0
    for state in range(len(policy)-1, -1, -1):
        h, w = shape
        x = int(state % w)
        if x == 0:
            if policy[state] == 0:
                count += 1
            else:
                return count
    return count


def find_likelihood(shape, policy, epsilon):
    safe_points = find_safe_points(shape, policy)
    right_path_chance = ((1-epsilon) + (epsilon/4))

    if safe_points < shape[0]-1:
        # we should care with the river when not using the bridge
        deterministic_path_length = 2 + (2*safe_points)
        likelihood_deterministic = pow(right_path_chance, deterministic_path_length)

        stochastic_path_length = shape[1] - 2
        likelihood_stochastic = pow(right_path_chance * 0.8, stochastic_path_length)

        return likelihood_deterministic * likelihood_stochastic
    else:
        right_path_length = shape[1] + (2*safe_points)
        return pow(right_path_chance, right_path_length)


def main(typeRunner):

    gamma = 0.99

    epsilon = 0.3

    threshold_likelihood = 0.2

    if typeRunner == 'QL':
        agents = [ag.AgentType.QL_TD, ag.AgentType.QL_TARGET]
        lambs = [(l - 15) / 10 for l in range(31)]
        lambs.remove(0.0)
        # lambs = [-0.3]
    else:
        agents = [ag.AgentType.DQN_TD_CACHE, ag.AgentType.DQN_TARGET_CACHE]
        # lambs = [-0.6, -0.5, -0.4, -0.3, -0.2, -0.1]
        lambs = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
        lambs = [-0.7, -0.8, -0.9, -1]
        lambs = [-1.1, -1.2, -1.3, -1.4, -1.5]
        # lambs = [(l-10)/10 for l in range(21)]
        # lambs = [(l - 15) / 10 for l in range(31)]
        # lambs.remove(0.0)

    # Building environment
    shape = (5, 4)
    env = RiverCrossingEnv(shape)

    alpha = 0.1

    # Building agents
    agents_models = {}
    for agent_type in agents:
        agents_models[agent_type] = {}

        # 1. Initialize the Target and Main models
        # Main Model (updated every step)
        model = ag.AgentModel.build(agent_type, env, alpha, gamma, -1)
        model.load_model()
        agents_models[agent_type]['MAIN'] = model

        # Target Model (updated every 100 steps)
        target_model = ag.AgentModel.build(agent_type, env, alpha, gamma, -1)
        target_model.set_weights(model)
        agents_models[agent_type]['TARGET'] = target_model

        # this will be used to refresh the agent when running second step
        fresh_model = ag.AgentModel.build(agent_type, env, alpha, gamma, -1)
        fresh_model.set_weights(model)
        agents_models[agent_type]['FRESH'] = fresh_model

    # collection value iteration policies
    policies = {}

    for lamb in lambs:
        policy, v, steps, updates, diffs, v_history = ValueIteration.run(env, lamb, gamma)
        policies[lamb] = policy

    sample = 0
    while True:
        sample += 1
        print('Sample', sample)
        for lamb in lambs:
            for agent_type in agents:

                # find agents models
                model = agents_models[agent_type]['MAIN']
                target_model = agents_models[agent_type]['TARGET']
                fresh_model = agents_models[agent_type]['FRESH']

                # reset them um fresh model
                model.set_weights(fresh_model)
                target_model.set_weights(fresh_model)
                model.reset(alpha, gamma, lamb, model.bellman_type)
                target_model.reset(alpha, gamma, lamb, target_model.bellman_type)

                # record start time
                t_0 = timeit.default_timer()

                # running
                try:
                    episodes = run(env, model, target_model, epsilon, policies[lamb], threshold_likelihood)
                except (ValueError, OverflowError):
                    episodes = 'ERROR'

                if episodes == 'ERROR' or episodes == 'ERROR_EXPLORATION':
                    print('RECREATE')
                    # 1. Initialize the Target and Main models
                    # Main Model (updated every step)
                    model = ag.AgentModel.build(agent_type, env, alpha, gamma, -1)
                    model.load_model()
                    agents_models[agent_type]['MAIN'] = model

                    # Target Model (updated every 100 steps)
                    target_model = ag.AgentModel.build(agent_type, env, alpha, gamma, -1)
                    target_model.set_weights(model)
                    agents_models[agent_type]['TARGET'] = target_model

                    # this will be used to refresh the agent when running second step
                    fresh_model = ag.AgentModel.build(agent_type, env, alpha, gamma, -1)
                    fresh_model.set_weights(model)
                    agents_models[agent_type]['FRESH'] = fresh_model


                # logging
                # calculate elapsed time and print
                t_1 = timeit.default_timer()
                elapsed_time = round((t_1 - t_0), 3)
                if not QUIET:
                    print(f"Elapsed time: {elapsed_time}s")

                out_log = "{}\t{}\t{}\t{}\t{}s\n".format(agent_type, gamma, lamb, episodes,elapsed_time)
                print(out_log)

                # write to result log
                file = open('DQNRunnerLikelihood_results_{}_neg_1_2.txt'.format(typeRunner), 'a')  # Open a file in append mode
                file.write(out_log)  # Write some text
                file.close()  # Close the file


if __name__ == "__main__":
    # venv/bin/python -m final.runner.DQNRunnerLikelihood QL
    # venv/bin/python -m final.runner.DQNRunnerLikelihood DQN
    args = sys.argv[1:]
    typeRunner = args[0]
    main(typeRunner)



