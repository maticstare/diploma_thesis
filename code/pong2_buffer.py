from pettingzoo.atari import pong_v3
import numpy as np
import pygame

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from keras.models import Sequential
from keras.layers import InputLayer, Dense
from keras.optimizers import Adam
from keras.models import clone_model

import time

def create_model():
    policy = Sequential()
    policy.add(InputLayer(input_shape=(4, )))
    policy.add(Dense(256, activation='relu'))
    policy.add(Dense(2048, activation='relu'))
    policy.add(Dense(256, activation='relu'))
    policy.add(Dense(3, activation='linear'))
    policy.compile(optimizer=Adam(learning_rate=0.00025), loss="mse", metrics=['mae'])
    target = clone_model(policy)
    return policy, target

def get_random_action():
    return np.random.randint(0, 3)

def get_policy_action(policy, state):
    q_values = policy.predict_on_batch(np.array([state])).flatten()
    action = np.argmax(q_values)
    return action

def get_state(observation):
    # my position, enemy position, ball x, ball y
    return np.array([observation[51]/300, observation[50]/300, observation[49]/300, observation[54]/300])

def get_result(observation):
    return int(observation[13]), int(observation[14])


def train(policy, target, experiences, exp_pointer, batch_size=32, gamma=0.99):
    if step < batch_size:
        return

    batch_state0, batch_action, batch_reward, batch_state1 = [], [], [], []
    for idx in np.random.randint(0, exp_pointer, batch_size)%memory_size:
        state0, action, reward, state1 = experiences[idx]
        batch_state0.append(state0)
        batch_action.append(action)
        batch_reward.append(reward)
        batch_state1.append(state1)
    batch_state0 = np.array(batch_state0)
    batch_action = np.array(batch_action)
    batch_reward = np.array(batch_reward)
    batch_state1 = np.array(batch_state1)

    policy_y = policy.predict_on_batch(batch_state0)
    target_y = target.predict_on_batch(batch_state1)

    max_q = np.amax(target_y, axis=1)
    batch_q = batch_reward + gamma * max_q

    for i in range(batch_size):
        policy_y[i][batch_action[i]] = batch_q[i]

    policy.train_on_batch(x=batch_state0, y=policy_y)

env = pong_v3.env(obs_type="ram")

action_map = [1, 2, 3] # fire, up, down


memory_size = 1000000

exp1_pointer = 0
exp2_pointer = 0

#p1 experiances
experiences1 = [(0,0,0,0), ] * memory_size

#p2 experiances
experiences2 = [(0,0,0,0), ] * memory_size

policy1, target1 = create_model()
policy2, target2 = create_model()

eps_max = 1.0
eps_min = 0.1
eps = eps_max
eps_steps = 1000000

episode = 1
episode_reward1 = 0
episode_reward2 = 0
episode_length = 0
episode_error = 0

env.reset()
observation, _, _, _, _ = env.last()

state0 = None
state1 = get_state(observation)

ball_dir = 0
new_ball_dir = 0
ball_dir_persistence = 0

steps = 10000000
target_update_steps = 5000
warmup_steps = 50000

step_by_length = []
episodes_in_epoch = 1
p1sum = 0
p2sum = 0
p1AvgReward = []
p2AvgReward = []

#count hits
p1_hits = 0
p2_hits = 0

p1_hits_arr = []
p2_hits_arr = []


for step in range(steps):
    state0 = state1

    # Decide the actions for both players
    if step >= warmup_steps and np.random.random() >= eps:
        action1 = get_policy_action(policy1, state0)
        action2 = get_policy_action(policy2, state0)
    else:
        action1 = get_random_action()
        action2 = get_random_action()

    # Player 1 (left)
    observation, reward1, terminated1, _, _ = env.last()
    if not terminated1:
        env.step(action_map[action1])

    # Player 2 (right)
    observation, reward2, terminated2, _, _ = env.last()
    if not terminated2:
        env.step(action_map[action2])

    terminated = terminated1 or terminated2

    # Observe state 1
    state1 = get_state(observation)

    # Compute the adjusted rewards.
    (_, _, x0, _) = state0
    (_, _, x1, _) = state1
    
    new_ball_dir_peek = np.sign(x1 - x0)
    if new_ball_dir_peek != 0:
        new_ball_dir = new_ball_dir_peek

    hit_reward1 = hit_reward2 = 0
    if ball_dir * new_ball_dir != 0 and new_ball_dir != ball_dir:
        if ball_dir_persistence > 10:
            if new_ball_dir > 0:
                hit_reward1 = 1
                p1_hits += 1
            else:
                hit_reward2 = 1
                p2_hits += 1

    if ball_dir * new_ball_dir != 0 and new_ball_dir == ball_dir:
        ball_dir_persistence += 1
    else:
        ball_dir_persistence = 0

    ball_dir = new_ball_dir

    if reward1 == 0:
        reward1 = hit_reward1
    else:
        reward1 *= 10
        new_ball_dir = 0
    
    if reward2 == 0:
        reward2 = hit_reward2
    else:
        reward2 *= 10
        new_ball_dir = 0

    episode_reward1 += reward1
    episode_reward2 += reward2
    episode_length += 1

    experiences1[exp1_pointer%memory_size] = (state0, action1, reward1, state1)
    experiences2[exp2_pointer%memory_size] = (state0, action2, reward2, state1)
    exp1_pointer += 1
    exp2_pointer += 1


    if step >= warmup_steps:
        # Player 1 train
        train(policy1, target1, experiences1, exp1_pointer)
        if step % target_update_steps == 0:
            target1.set_weights(policy1.get_weights())
        
        # Player 2 train
        train(policy2, target2, experiences2, exp2_pointer)
        if step % target_update_steps == 0:
            target2.set_weights(policy2.get_weights())

    if terminated:
        observation, _, _, _, _ = env.last()
        p1Score, p2Score = get_result(observation)

        episode_reward2 = p2Score - p1Score
        episode_reward1 = p1Score - p2Score
        
        p1sum += episode_reward1
        p2sum += episode_reward2

        episodes_in_epoch += 1

        step_by_length += [(step, episode_length)]
        print(step, episode_length)

        #ball hits
        p1_hits_arr += [(episode, p1_hits)]
        p2_hits_arr += [(episode, p2_hits)]

        p1_hits = 0
        p2_hits = 0

        episode += 1
        episode_reward1 = 0
        episode_reward2 = 0
        episode_length = 0
        episode_error = 0

        ball_dir = 0
        new_ball_dir = 0
        ball_dir_persistence = 0

        if step < eps_steps:
            eps = eps_max - step/eps_steps * (eps_max - eps_min)
        else:
            eps = eps_min
        
        env.reset()
        observation, _, _, _, _ = env.last()
        state0 = None
        state1 = get_state(observation)
    
    if step % 1000000 == 999999:
        policy1.save(f"p1_1M_256_2048_256_{step+1}")
        policy2.save(f"p2_1M_256_2048_256_{step+1}")
        
        f = open("1M_256_2048_256.txt", "w")
        f.write(f"Average reward:\n{p1AvgReward}\n{p2AvgReward}\n\nStep by length:\n{step_by_length}\n\nHits per episode:\n{p1_hits_arr}\n{p2_hits_arr}")
        f.close()

    if step % 100000 == 99999:
        p1AvgReward += [(step+1, p1sum/episodes_in_epoch)]
        p2AvgReward += [(step+1, p2sum/episodes_in_epoch)]
        p1sum, p2sum, episodes_in_epoch = 0, 0, 1

env.close()
