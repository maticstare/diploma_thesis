import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # Do not print Tensorflow debug messages.

from keras.models import load_model
from pettingzoo.atari import pong_v3
import pygame
import numpy as np
import random
import tensorflow as tf
import time


RANDOM_ACTIONS_PERCENT = 0.1

model1 = load_model("koda/mojaKoda/p1_exp1M_3000000")
model2 = load_model("koda/mojaKoda/p2_exp1M_3000000")


def get_policy_action(model, state):
    q_values = model.predict_on_batch(np.array([state])).flatten()
    action = np.argmax(q_values)
    return action

def get_state(observation):
    # my position, enemy position, ball x, ball y
    return np.array([observation[51]/300, observation[50]/300, observation[49]/300, observation[54]/300])

def get_random_action():
    return np.random.randint(0, 2)

#prepare environment
pygame.init()
env = pong_v3.env(render_mode='human', obs_type="ram")
env.reset()
done = False
action_map = [2, 3]


while not done:
    pygame.event.get()
    env.render()

    observation, _, done, _, _ = env.last()
    state = get_state(observation)

    #p1 action (orange)
    if state[2] <= 13/75 and state[3] == 0: #do this to serve the goal after scoring  
        env.step(1)
    else:
        if np.random.random() >= RANDOM_ACTIONS_PERCENT:
            action = get_policy_action(model1, state)
        else:
            action = get_random_action()
        env.step(action_map[action])

    
    observation, _, done, _, _ = env.last()
    state = get_state(observation)

    #p2 action (green)
    if state[2] >= 41/60 and state[3] == 0: #do this to serve the goal after scoring     
        env.step(1)
    else:
        if np.random.random() >= RANDOM_ACTIONS_PERCENT:
            action = get_policy_action(model2, state)
        else:
            action = get_random_action()
        env.step(action_map[action])

    time.sleep(0.001)

env.close()