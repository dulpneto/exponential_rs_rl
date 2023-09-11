from abc import ABC, abstractmethod

from agent import BellmanUpdate as bu

import tensorflow as tf
import numpy as np
from tensorflow import keras
from copy import copy
from collections import defaultdict
from collections import deque

import math

import pickle

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


# Types of agents implemented
class AgentType:
    # Q-Learning
    QL = 'QL'
    # Deep QNetwork
    DQN = 'DQN'
    DQN_CACHED = 'DQN_CACHED'
    DQN_SKIP = 'DQN_SKIP'
    # Deep QNetwork with convolutional network to handle states as images
    DQN_CONV = 'DQN_CONV'
    DQN_CONV_CACHED = 'DQN_CONV_CACHED'
    DQN_CONV_SKIP = 'DQN_CONV_SKIP'


# Agent default behaviour
class AgentModel(ABC):

    def __init__(self, env, alpha, gamma, lamb, bellman):
        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        self.lamb = lamb
        self.bellman = bellman
        self.create_model()

    def reset(self, alpha, gamma, lamb):
        self.alpha = alpha
        self.gamma = gamma
        self.lamb = lamb
        self.bellman.reset()

    @abstractmethod
    def create_model(self):
        pass

    @abstractmethod
    def load_model(self):
        pass

    @abstractmethod
    def update_model(self, mini_batch, target_model):
        pass

    @abstractmethod
    def set_weights(self, model):
        pass

    @abstractmethod
    def save_model(self, name):
        pass

    @abstractmethod
    def find_qs(self, state):
        pass

    @abstractmethod
    def print_qs_model(self):
        pass

    def find_safe_points(self):
        count = 0
        h, w = self.env.shape
        for state in range((h * w) - 1, -1, -1):
            h, w = self.env.shape
            x = int(state % w)

            if self.env.state_as_img:
                state_formated = self.env.state_img_cache[state]
            else:
                state_formated = state

            if x == 0:
                if np.argmax(self.find_qs(state_formated)) == 0:
                    count += 1
                else:
                    return count
        return count

    @staticmethod
    def build(type, bellman_type, env, alpha, gamma, lamb):

        bellman_update = bu.BellmanUpdate.build(bellman_type, alpha, gamma, lamb, env.max_abs_r)

        if type == AgentType.QL:
            return QLearningModel(env, alpha, gamma, lamb, bellman_update)
        elif type == AgentType.DQN:
            return DQNModel(env, alpha, gamma, lamb, bellman_update)
        elif type == AgentType.DQN_CACHED:
            return DQNModelCached(env, alpha, gamma, lamb, bellman_update)
        elif type == AgentType.DQN_SKIP:
            return DQNSkipModelUsage(env, alpha, gamma, lamb, bellman_update)
        elif type == AgentType.DQN_CONV:
            return DQNConvModel(env, alpha, gamma, lamb, bellman_update)
        elif type == AgentType.DQN_CONV_CACHED:
            return DQNConvModelCached(env, alpha, gamma, lamb, bellman_update)
        elif type == AgentType.DQN_CONV_SKIP:
            return DQNConvSkipModelUsage(env, alpha, gamma, lamb, bellman_update)
        else:
            raise Exception("Not implemented {}".format(type))
        return


# QLearning
class QLearningModel(AgentModel):

    def create_model(self):
        self.Q = defaultdict(lambda: np.ones(self.env.action_space.n) * self.bellman.default_V)

    def load_model(self):
        # do nothing
        return

    def update_model(self, mini_batch, target_model):
        for index, (current_state, action, reward, next_state, done) in enumerate(mini_batch):
            future_qs = target_model.Q[next_state]
            current_qs = self.Q[current_state]
            self.Q[current_state][action] = self.bellman.bellman_update(current_qs, future_qs, action, reward)

    def set_weights(self, model):
        # deep copy Q
        self.Q = copy(model.Q)

    def save_model(self, name):
        with open(name, 'wb') as fp:
            pickle.dump(self.Q, fp)

    def find_qs(self, state):
        # Q[s]
        return self.Q[state]

    def print_qs_model(self):
        pass


# DQN with original implementation without cache strategy
class DQNModel(AgentModel):

    def create_model(self):
        state_shape = self.env.observation_space.shape
        action_shape = self.env.action_space.n

        init = tf.keras.initializers.HeUniform()
        self.model = keras.Sequential()
        self.model.add(keras.layers.Dense(24, input_shape=state_shape, activation='relu', kernel_initializer=init))
        self.model.add(keras.layers.Dense(12, activation='relu', kernel_initializer=init))
        self.model.add(keras.layers.Dense(action_shape, activation='linear', bias_initializer=tf.keras.initializers.Constant(0.0)))
        self.model.compile(loss=tf.keras.losses.Huber(),
                           optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), metrics=['accuracy'])

        self.cached_q = False

    def load_model(self):
        # treinar ou nao?
        return

    def update_model(self, mini_batch, target_model):

        current_states = np.array(
            [self.encode_observation(transition[0]) for transition in mini_batch])
        current_qs_list = self.model.predict(current_states, verbose=0)
        new_current_states = np.array(
            [self.encode_observation(transition[3]) for transition in mini_batch])
        future_qs_list = target_model.model.predict(new_current_states, verbose=0)

        x = []
        y = []
        for index, (current_state, action, reward, next_state, done) in enumerate(mini_batch):
            future_qs = future_qs_list[index]
            current_qs = current_qs_list[index]

            current_qs[action] = self.bellman.bellman_update(current_qs, future_qs, action, reward)

            x.append(self.encode_observation(current_state))
            y.append(current_qs)

        self.model.fit(np.array(x), np.array(y), batch_size=len(mini_batch), verbose=0, shuffle=True)

    def set_weights(self, model):
        self.model.set_weights(model.model.get_weights())

    def save_model(self, name):
        self.model.save_weights(name)

    def find_qs(self, state):
        encoded = self.encode_observation(state)
        encoded_reshaped = encoded.reshape([1, encoded.shape[0]])
        predicted = self.model.predict(encoded_reshaped, verbose=0).flatten()

        return predicted

    def print_qs_model(self):
        pass

    def encode_observation(self, state):
        encoded = np.zeros(self.env.observation_space.shape)
        encoded[int(state)] = 1
        return encoded


# DQN with Cache strategy to make tests run faster
class DQNModelCached(DQNModel):

    def update_model(self, mini_batch, target_model):

        for index, (current_state, action, reward, next_state, done) in enumerate(mini_batch):
            future_qs = target_model.find_qs(next_state)
            current_qs = self.find_qs(current_state)

            for q in current_qs:
                if q == float('inf') or q == float('-inf'):
                    print('INFINITO')
                    self.cached_q = False
                    self.find_qs(current_state)

            self.Q[current_state][action] = self.bellman.bellman_update(current_qs, future_qs, action, reward)

        x = [self.encode_observation(s) for s in range(self.env.observation_space.shape[0])]
        y = [self.bellman.bellman_normalize(self.Q[s]) for s in range(self.env.observation_space.shape[0])]

        self.model.fit(np.array(x), np.array(y), batch_size=len(mini_batch), verbose=0, shuffle=True)

        # marking cache to be refreshed
        # self.cached_q = False

    def set_weights(self, model):
        # WITH MODEL USAGE
        self.cached_q = False
        self.model.set_weights(model.model.get_weights())

    def find_qs(self, state):

        # making a batch prediction to speed up tests
        if not self.cached_q:
            # WITH MODEL USAGE
            x = [self.encode_observation(s) for s in range(self.env.observation_space.shape[0])]
            prediction = self.model.predict(np.array(x), batch_size=len(x), verbose=0)
            self.Q = [self.bellman.bellman_denormalize(qs) for qs in prediction]

            for q_S in self.Q:
                for q in q_S:
                    if q == float('inf') or  q == float('-inf'):
                        print('INFINITO')
                    if math.isnan(q):
                        print('NAN')

            self.cached_q = True

        return self.Q[state]

    def print_qs_model(self):

        print('MODEL VALUES')

        h, w = self.env.shape
        lineState = ''
        for y in range(h):
            for x in range(w):
                state = x + (y * w)
                encoded = self.encode_observation(state)
                encoded_reshaped = encoded.reshape([1, encoded.shape[0]])
                q_s = self.model.predict(encoded_reshaped, verbose=0).flatten()
                lineState = '{}\t{}'.format(lineState, str(round(max(q_s), 3)))
            lineState = '{}\n'.format(lineState)
        print('')
        print(lineState)


# DQN that skip using model - use only Q table but model is trained
class DQNSkipModelUsage(DQNModelCached):

    def set_weights(self, model):
        # SKIP MODEL USAGE
        if not model.cached_q:
            self.Q = defaultdict(lambda: np.ones(self.env.action_space.n) * self.bellman.default_V)
        else:
            self.Q = model.Q

    def find_qs(self, state):

        # making a batch prediction to speed up tests
        if not self.cached_q:
            # SKIP MODEL USAGE
            self.Q = defaultdict(lambda: np.ones(self.env.action_space.n) * self.bellman.default_V)

            self.cached_q = True

        return self.Q[state]


# DQN with convolutional network to handle states as images
class DQNConvModel(DQNModel):

    def create_model(self):
        # keep a sing image on stack since we are not working object velocity
        self.frame_stack_num = 1

        action_shape = self.env.action_space.n

        # Neural Net for Deep-Q learning Model
        self.model = keras.Sequential()
        self.model.add(keras.layers.Conv2D(filters=6, kernel_size=(7, 7), strides=3, activation='relu',
                         input_shape=(100, 100, 1)))
        self.model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
        self.model.add(keras.layers.Conv2D(filters=12, kernel_size=(4, 4), activation='relu'))
        self.model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
        self.model.add(keras.layers.Flatten())
        self.model.add(keras.layers.Dense(216, activation='relu'))
        self.model.add(keras.layers.Dense(4, activation=None))
        self.model.compile(loss='mean_squared_error',
                           optimizer=tf.keras.optimizers.Adam(learning_rate=0.001, epsilon=1e-7))

        self.cached_q = False

    def find_qs(self, state):
        encoded = self.encode_observation(state)
        predicted = self.model.predict(np.expand_dims(encoded, axis=0), verbose=0).flatten()
        return predicted

    def encode_observation(self, state):
        frame_stack = deque([state]*self.frame_stack_num, maxlen=self.frame_stack_num)
        return np.transpose(frame_stack, (1, 2, 0))

# DQN with convolutional network and with Cache strategy to make tests run faster
class DQNConvModelCached(DQNConvModel):

    def update_model(self, mini_batch, target_model):

        for index, (current_state, action, reward, next_state, done) in enumerate(mini_batch):
            future_qs = target_model.find_qs(next_state)
            current_qs = self.find_qs(current_state)

            current_state_idx = self.env.find_s_from_img(current_state)
            self.Q[current_state_idx][action] = self.bellman.bellman_update(current_qs, future_qs, action, reward)

        x = [self.encode_observation(self.env.state_img_cache[s]) for s in range(self.env.observation_space.shape[0])]
        y = [self.bellman.bellman_normalize(self.Q[s]) for s in range(self.env.observation_space.shape[0])]

        self.model.fit(np.array(x), np.array(y), batch_size=len(mini_batch), verbose=0, shuffle=True)

        # marking cache to be refreshed
        # self.cached_q = False

    def set_weights(self, model):
        # WITH MODEL USAGE
        self.cached_q = False
        self.model.set_weights(model.model.get_weights())

    def find_qs(self, state):

        # making a batch prediction to speed up tests
        if not self.cached_q:
            # WITH MODEL USAGE
            x = [self.encode_observation(self.env.state_img_cache[s]) for s in range(self.env.observation_space.shape[0])]
            prediction = self.model.predict(np.array(x), batch_size=len(x), verbose=0)
            self.Q = [self.bellman.bellman_denormalize(qs) for qs in prediction]

            self.cached_q = True

        state_idx = self.env.find_s_from_img(state)
        return self.Q[state_idx]

    def print_qs_model(self):

        print('MODEL VALUES')

        h, w = self.env.shape
        lineState = ''
        for y in range(h):
            for x in range(w):
                state = x + (y * w)
                state_img = self.env.state_img_cache[state]
                encoded = self.encode_observation(state_img)
                q_s = self.model.predict(np.expand_dims(encoded, axis=0), verbose=0).flatten()
                lineState = '{}\t{}'.format(lineState, str(round(max(q_s), 3)))
            lineState = '{}\n'.format(lineState)
        print('')
        print(lineState)


# DQN that skip using model - use only Q table but model is trained
class DQNConvSkipModelUsage(DQNConvModelCached):

    def set_weights(self, model):
        # SKIP MODEL USAGE
        if not model.cached_q:
            self.Q = defaultdict(lambda: np.ones(self.env.action_space.n) * self.bellman.default_V)
        else:
            self.Q = model.Q

    def find_qs(self, state):

        # making a batch prediction to speed up tests
        if not self.cached_q:
            # SKIP MODEL USAGE
            self.Q = defaultdict(lambda: np.ones(self.env.action_space.n) * self.bellman.default_V)

            self.cached_q = True

        state_idx = self.env.find_s_from_img(state)
        return self.Q[state_idx]
