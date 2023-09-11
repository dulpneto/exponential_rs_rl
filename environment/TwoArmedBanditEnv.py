import gym
from gym import spaces
import matplotlib.pyplot as plt
import numpy as np
import random
import math

# number of action - deterministic arm or stochastic
N_DISCRETE_ACTIONS = 2


# Two-armed Bandit where
#  - Arm 0 is a discrete arm that returns the given reward
#  - Arm 1 is a stochastic arm that returns a sample from a Gaussian distribution from given mean and precision
# States defined as initial(0) and final (1)
# Actions defined as choosing arm 0 and choosing arm 1
class TwoArmedBandit(gym.Env):
    """Custom Environment that follows gym interface"""
    metadata = {'render.modes': ['human']}

    def __init__(self, r_0, mu, p=2, plot=False):
        # super(RiverCrossingEnv, self).__init__()
        # Define action and observation space
        # They must be gym.spaces objects
        self.action_space = spaces.Discrete(N_DISCRETE_ACTIONS)

        self.s0 = 0
        self.current_s = self.s0

        # terminals
        self.G = []
        self.G.append(1)

        # arm 0 setup
        self.r_0 = r_0

        # arm 1 setup
        self.mu = mu  # mean
        self.p = p  # precision
        self.sigma = 1 / math.sqrt(p)  # standard deviation
        # self.sigma = 1/p # standard deviation

        # self.observation_space = spaces.Discrete(w * h)
        self.observation_space = spaces.Box(low=0, high=1, shape=(N_DISCRETE_ACTIONS,))

        # probabilities - (state,action)[(state_next, probability,reward),...]
        self.P = {}

        self.P[(0, 0)] = [(1, 1, self.r_0)]
        self.P[(0, 1)] = [(1, 0.25, (self.mu - (self.p / 2))),
                          (1, 0.5, (self.mu)),
                          (1, 0.25, (self.mu + (self.p / 2)))]

        self.P[(1, 0)] = [(1, 1, 0)]
        self.P[(1, 1)] = [(1, 1, 0)]

        # max_abs_r = sup(|r|)
        self.max_abs_r = max(abs(self.r_0), abs(self.mu - (self.p / 2)), abs(self.mu + (self.p / 2)))

        if plot:
            s = np.random.normal(self.mu, self.sigma, 10_000)
            plt.hist(s, bins=50)
            plt.gca().set(title='Arm 1', ylabel='Frequency');

    def step(self, action):
        done = True

        random_number = random.uniform(0, 1)
        t_sum = 0.0
        for s_next, t, r in self.P[(self.current_s, action)]:
            t_sum += t
            if random_number <= t_sum:
                self.current_s = s_next
                return s_next, r, done, {}

    def reset(self):
        # Reset the state of the environment to an initial state
        self.current_s = self.s0
        return self.current_s

    def render(self, mode='human', close=False):
        # Render the environment to the screen
        return self.current_s