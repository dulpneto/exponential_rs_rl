import sys

import tensorflow as tf
import numpy as np

import cv2

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

def main():
    # Building environment
    shape = (10, 10)
    h, w = shape
    env = RiverCrossingEnv(shape, state_as_img=True, state_img_width=500)

    #print(env.reset())
    RiverCrossingEnv.draw_img_state(shape, policy=[], state_img_width=500, state=-99)

    for safe_steps in range (1,10):

        policy = np.ones(h*w) * -1
        margin = 90
        for s in range(safe_steps):
            policy[margin] = 0
            margin -= 10

        start = margin
        for i in range(start,start+9):
            policy[i] = 2

        margin += 9
        for s in range(safe_steps):
            policy[margin] = 1
            margin += 10
        RiverCrossingEnv.draw_img_state(shape, policy=policy, state_img_width=500, state=(safe_steps * -1))

    if True:
        return

    data = ''
    gamma = 0.99
    for lamb in [-2.0, -1.0, -0.75, -0.5, -0.2, -0.1, 0.1, 0.2, 0.5, 0.75, 1.0, 2.0]:
        policy, v, steps, updates, diffs, v_history = ValueIteration.run(env, lamb, gamma)
        safe_points = ValueIteration.find_safe_points(env, policy)
        data+='{},'.format(safe_points)

    print(data)

    h, w = shape
    for y in range(h):
        for x in range(w):
            s = x + (y * w)
            #RiverCrossingEnv.draw_img_state(shape, s)
    #process_state_image(0)

def process_state_image(state):
    image = cv2.imread('environment/img/river_{}.png'.format(state))
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cv2.imwrite('environment/img/river_{}_2.png'.format(state), image_gray)
    image_gray = image_gray.astype(float)
    image_gray /= 255.0

    image = image.astype(float)
    image /= 255.0

    for i in range(len(image)):
        for j in range(len(image[i])):
            #for c in range(len(image[i][j])):
            print(image[i][j], image_gray[i][j])

    return image_gray


if __name__ == "__main__":
    main()



