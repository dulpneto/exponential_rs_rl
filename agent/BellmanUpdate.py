from abc import ABC, abstractmethod

import numpy as np
import math
import sys
from agent.utils import UnderflowError

UNDERFLOW_THRESHOLD = 100


class Type:
    TARGET = 'Target'
    TD = 'TD'
    TD_TRUNC = 'TD_TRUNC'
    SOFT_INDICATOR = 'SI'
    TARGET_LOG_SUM_EXP = 'LSE'


class BellmanUpdate(ABC):

    def __init__(self, alpha, gamma, lamb, max_r):
        self.alpha = alpha
        self.gamma = gamma
        self.lamb = lamb
        self.default_V = self.init_default_v()
        self.max_u = 0
        self.min_u = sys.maxsize
        self.under_count = 0
        self.not_under_count = 0

    @abstractmethod
    def init_default_v(self):
        pass

    def bellman_update(self, current_qs, future_qs, action, reward):
        if self.lamb == 0:
            return self.bellman_update_neutral(current_qs, future_qs, action, reward)
        else:
            return self.bellman_update_risk(current_qs, future_qs, action, reward)

    def bellman_update_neutral(self, current_qs, future_qs, action, reward):
        v = np.max(future_qs)
        target = reward + (self.gamma * v)
        td = target - current_qs[action]
        return current_qs[action] + (self.alpha * td)

    def reset(self):
        self.default_V = self.init_default_v()

    @abstractmethod
    def bellman_update_risk(self, current_qs, future_qs, action, reward):
        pass

    @abstractmethod
    def bellman_normalize(self, current_qs):
        pass

    @abstractmethod
    def bellman_denormalize(self, current_qs):
        pass

    def utility(self, x):
        return np.sign(self.lamb) * math.exp(self.lamb * x)

    def reverse_utility(self, x):
        return math.log(np.sign(self.lamb) * x) / self.lamb

    def safe_max(self, qs):
        v = np.max(qs)
        return self.validate_v(v)

    def safe_q(self, qs, action):
        v = qs[action]
        return self.validate_v(v)

    @abstractmethod
    def validate_v(self, v):
        pass

    @staticmethod
    def build(type, alpha, gamma, lamb, max_abs_r):
        if type == Type.TARGET:
            return TargetUpdate(alpha, gamma, lamb, max_abs_r)
        elif type == Type.TD:
            return TDUpdate(alpha, gamma, lamb, max_abs_r)
        elif type == Type.TD_TRUNC:
            return TDTruncUpdate(alpha, gamma, lamb, max_abs_r)
        elif type == Type.TARGET_LOG_SUM_EXP:
            return TargetLogSumExpUpdate(alpha, gamma, lamb, max_abs_r)
        elif type == Type.SOFT_INDICATOR:
            return SoftIndicatorUpdate(alpha, gamma, lamb, max_abs_r)
        else:
            raise Exception("Not implemented {}".format(type))


# Defining Bellman update to the Target Value
class TargetUpdate(BellmanUpdate):

    def init_default_v(self):
        # return 1 * np.sign(self.lamb)
        return 0

    def bellman_update_risk(self, current_qs, future_qs, action, reward):

        # considering q as on log domain so we should apply utility on q to make it right
        v = self.safe_max(future_qs)
        target = reward + (self.gamma * v)
        u = self.utility(target)

        if u == 0.0:
            self.under_count += 1
        else:
            self.not_under_count += 1

        #if self.under_count >= UNDERFLOW_THRESHOLD:
        #    raise UnderflowError('Underflow on {}'.format(target))

        if reward != 0.0 and abs(u) > self.max_u:
            self.max_u = abs(u)

        if reward != 0.0 and abs(u) < self.min_u:
            self.min_u = abs(u)

        u_q = self.utility(self.safe_q(current_qs, action))
        return self.reverse_utility(u_q + (self.alpha * (u - u_q)))

    def bellman_normalize(self, current_qs):
        # return [math.log(np.sign(self.lamb) * self.validate_v(q)) / self.lamb for q in current_qs]
        return current_qs

    def bellman_denormalize(self, current_qs):
        # return [self.validate_v(np.sign(self.lamb) * math.exp(self.lamb * q)) for q in current_qs]
        return current_qs

    # handling invalid values to avoid braking log and exp
    # this help to not initialize the network
    def validate_v(self, v):
        # if self.lamb > 0 and v <= 0:
        #    return self.default_V
        # elif self.lamb < 0 and v > self.default_V:
        #     return self.default_V
        #if v > 0:
        #    return 0
        return v


# Defining Bellman update to the Target Value using Log Sum Exp
class TargetLogSumExpUpdate(BellmanUpdate):

    def init_default_v(self):
        # return 1 * np.sign(self.lamb)
        return 0

    def bellman_update_risk(self, current_qs, future_qs, action, reward):
        # considering q as on log domain so we should apply utility on q to make it right
        target = reward + self.gamma * self.safe_max(future_qs)

        exp_1 = self.lamb * current_qs[action] + math.log(1 - self.alpha)
        exp_2 = self.lamb * target + math.log(self.alpha)

        a = max([exp_1, exp_2])
        b = min([exp_1, exp_2])

        return (a + math.log(1 + math.exp(b - a))) / self.lamb

    def bellman_normalize(self, current_qs):
        # return [math.log(np.sign(self.lamb) * self.validate_v(q)) / self.lamb for q in current_qs]
        return current_qs

    def bellman_denormalize(self, current_qs):
        # return [self.validate_v(np.sign(self.lamb) * math.exp(self.lamb * q)) for q in current_qs]
        return current_qs

    # handling invalid values to avoid braking log and exp
    # this help to not initialize the network
    def validate_v(self, v):
        # if self.lamb > 0 and v <= 0:
        #    return self.default_V
        # elif self.lamb < 0 and v > self.default_V:
        #     return self.default_V
        #if v > 0:
        #    return 0
        return v


# Defining Bellman update to the Tempora Difference Value
class TDUpdate(BellmanUpdate):

    def __init__(self, alpha, gamma, lamb, max_r):
        self.alpha = alpha
        self.gamma = gamma
        self.lamb = lamb
        self.default_V = self.init_default_v()
        self.max_u = 0
        self.min_u = sys.maxsize
        self.under_count = 0
        self.y_0 = 0
        self.x_0 = np.sign(self.lamb) * math.exp(self.lamb * self.y_0)

    def init_default_v(self):
        return 0

    def bellman_update_risk(self, current_qs, future_qs, action, reward):
        v = self.safe_max(future_qs)
        td = reward + (self.gamma * v) - self.safe_q(current_qs, action)
        u = self.utility(td)

        if u == 0.0:
            self.under_count += 1
        else:
            self.under_count -= 1

        #if self.under_count >= UNDERFLOW_THRESHOLD:
        #    raise UnderflowError('Underflow on {}'.format(td))

        if reward != 0.0 and abs(u) > self.max_u:
            self.max_u = abs(u)

        if reward != 0.0 and abs(u) < self.min_u:
            self.min_u = abs(u)

        return self.safe_q(current_qs, action) + (self.alpha * (u - self.x_0))

    def bellman_normalize(self, current_qs):
        return current_qs

    def bellman_denormalize(self, current_qs):
        return current_qs

    # handling invalid values to avoid braking log and exp
    # this help to not initialize the network
    def validate_v(self, v):
        #if v > 0:
        #    return 0
        return v

class TDTruncUpdate(BellmanUpdate):

    def __init__(self, alpha, gamma, lamb, max_abs_r):
        self.alpha = alpha
        self.gamma = gamma
        self.lamb = lamb
        self.max_abs_r = max_abs_r
        self.default_V = self.init_default_v()
        self.max_u = 0
        self.min_u = sys.maxsize
        self.under_count = 0
        self.y_0 = 0
        self.x_0 = np.sign(self.lamb) * math.exp(self.lamb * self.y_0)
        self.l_x = self.lower_x()
        self.s_x = self.sup_x()
        self.div_const = 1
        self.const = self.find_const(self.l_x, self.s_x) / self.div_const

        np.seterr(all='raise')

    def init_default_v(self):
        return 0

    def calc_const(self, x, y):
        print(x, y, x-y)
        return (self.utility(x) - self.utility(y)) / (x - y)

    def find_const(self, l_x, s_x):
        # diff lower than 1e-13 will make (x-y) equals 0
        const_diff = 1e-11

        # if l_x less or equals -715 and s_x equals or greater 715 const will be 0

        if self.lamb > 0:
            x = l_x
            y = l_x + const_diff
        else:
            x = s_x - const_diff
            y = s_x

        return self.calc_const(x, y)

    def lower_x(self):
        return self.y_0 - ((2 * self.max_abs_r) / (1 - self.gamma))

    def sup_x(self):
        return self.y_0 + ((2 * self.max_abs_r) / (1 - self.gamma))

    def bellman_update_risk(self, current_qs, future_qs, action, reward):
        v = self.safe_max(future_qs)
        td = reward + (self.gamma * v) - self.safe_q(current_qs, action)

        if td <= self.l_x:
            u = self.utility(self.l_x) + (self.const * (td - self.l_x))
        elif td >= self.s_x:
            u = self.utility(self.s_x) + (self.const * (td - self.s_x))
        else:
            u = self.utility(td)

        if u == 0.0:
            self.under_count += 1
        else:
            self.under_count -= 1

        #if self.under_count >= UNDERFLOW_THRESHOLD:
        #    raise UnderflowError('Underflow on {}'.format(td))

        if reward != 0.0 and abs(u) > self.max_u:
            self.max_u = abs(u)

        if reward != 0.0 and abs(u) < self.min_u:
            self.min_u = abs(u)

        result = self.safe_q(current_qs, action) + (self.alpha * (u - self.x_0))

        if result == float('inf') or result == float('-inf'):
            print('INFINITO')

        return result

    def bellman_normalize(self, current_qs):
        return current_qs

    def bellman_denormalize(self, current_qs):
        return current_qs

    # handling invalid values to avoid braking log and exp
    # this help to not initialize the network
    def validate_v(self, v):
        #if v > 0:
        #    return 0
        return v


# Defining Bellman update to the Tempora Difference Value using soft indicator
class SoftIndicatorUpdate(BellmanUpdate):

        def __init__(self, alpha, gamma, lamb, max_r):
            self.alpha = alpha
            self.gamma = gamma
            self.lamb = lamb
            self.default_V = self.init_default_v()
            self.max_u = 0
            self.min_u = sys.maxsize
            self.under_count = 0
            self.y_0 = 0
            self.x_0 = 0

        def init_default_v(self):
            return 0

        def bellman_update_risk(self, current_qs, future_qs, action, reward):
            v = self.safe_max(future_qs)
            td = reward + (self.gamma * v) - self.safe_q(current_qs, action)
            u = ((2 * td) / (1 + math.exp(-self.lamb * td)))

            if u == 0.0:
                self.under_count += 1
            else:
                self.under_count -= 1

            #if self.under_count >= UNDERFLOW_THRESHOLD:
            #    raise UnderflowError('Underflow on {}'.format(-self.lamb * td))


            if reward != 0.0 and abs(u) > self.max_u:
                self.max_u = abs(u)

            if reward != 0.0 and abs(u) < self.min_u:
                self.min_u = abs(u)

            return self.safe_q(current_qs, action) + (self.alpha * (u - self.x_0))

        def bellman_normalize(self, current_qs):
            return current_qs

        def bellman_denormalize(self, current_qs):
            return current_qs

        # handling invalid values to avoid braking log and exp
        # this help to not initialize the network
        def validate_v(self, v):
            #if v > 0:
            #    return 0
            return v

