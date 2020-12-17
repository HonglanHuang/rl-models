import os
import time
import logging
logging.basicConfig(level=logging.INFO)

import numpy as np
import random
from collections import deque, namedtuple
from itertools import product

# env
import gym

# rl model
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, ReLU, Input, concatenate
# from tensorflow.keras.losses import MeanSquaredError

# OU Noise

from ddpg import *
from OU_noise import OUNoise

# hyper
LEGAL_ACTION_CONSTRAINT = False

############################## WOLP Agent #############################
# define transition named tuple
Transitions = namedtuple('Transitions', ['state', 'action', 'reward', 'next_state', 'done'])

class wolp_agent(DDPG):
    def __init__(self,
                 state_dim,
                 action_dim,
                 action_range,
                 k_candidates,
                 lr_a,
                 lr_c,
                 buffer_size,
                 batch_size,
                 gamma,
                 tau,
                 save_graph):
        super().__init__(state_dim,
                 action_dim,
                 action_range,
                 lr_a,
                 lr_c,
                 buffer_size,
                 batch_size,
                 gamma,
                 tau,
                 save_graph)
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_range = action_range
        self.k_candidates = k_candidates
        self.lr_a = lr_a
        self.lr_c = lr_c
        self.gamma = gamma
        self.buffer_size = int(buffer_size)
        self.batch_size = int(batch_size)
        self.tau = tau

        # tf session
        self.sess = tf.Session()

        # initialize variables
        self.sess.run(tf.global_variables_initializer())
        # save graoh
        if save_graph:
            tf.summary.FileWriter('./logs/', self.sess.graph)

    def select_action(self, state):
        proto_action = super().act(state)
        # print(proto_action)

        # epsilon-greedy
        if np.random.uniform() < self.eps:
            action = np.random.choice([valid for valid in range(state.shape[0]) if state[valid] > 0], self.action_dim, replace=False)
        else:
            action = self.refine_action(state, proto_action, self.k_candidates, target=False)

        try:
            action = action.flatten()  # want final action to be 1d
        except:
            pass

        return np.array(action)

    def refine_action(self, state, proto_action, k_candidates, target):
        if len(proto_action.shape) > 1:
            proto_action = proto_action.flatten()

        # to be updated
        if LEGAL_ACTION_CONSTRAINT:
            available_actions = [x for x in range(len(env.legal_pipe_name_list)) if state[x] > 0]  # available actions at current state
        else:
            available_actions = np.arange(env.action_space.n).tolist()

        # for idx in range(k_candidates, 1, -1):
        while True:
            if len(available_actions) < k_candidates * proto_action.shape[0]:
                self.k_candidates -= 1
            else:
                break

        nearest_actions = {}
        for d in range(proto_action.shape[0]):
            # find k nearest available actions in dth dimension
            try:
                nearest_actions_ids = np.argsort(np.abs(proto_action[d] - np.asarray(available_actions)))[:k_candidates]   # idx of distance from small to large
            except:
                nearest_actions_ids = np.argsort(np.abs(proto_action[d] - np.asarray(available_actions)))  # when nearest actions < k candidates, just take all nearest actions

            nearest_actions[d] = [available_actions[i] for i in nearest_actions_ids]

            # update available actions
            [available_actions.remove(x) for x in list(nearest_actions[d])]

        cand_actions = np.array(list(product(*nearest_actions.values())))  # find the power set of the candidate actions

        states = np.tile(state, [cand_actions.shape[0], 1])

        if target:
            qs = self.Target_Critic.forward(states, cand_actions)  # evaluate by target critic
        else:
            qs = self.Critic.forward(states, cand_actions)

        idx_max = np.argmax(qs, axis=0)

        best_action = cand_actions[idx_max]

        # return best_action.item()
        return best_action.flatten()

    def learn(self):
        states, actions, rewards, next_states, dones = super.Buffer.sample(self.batch_size)

        # get next actions from target actor
        next_proto_actions = self.Target_Actor.forward(next_states)  # batch_size x action_dim

        # actions at next state are generated from full target policy (refined)
        next_actions = next_proto_actions.copy()
        for idx in range(next_proto_actions.shape[0]):
            next_actions[idx, :] = self.refine_action(next_states[idx, :], next_proto_actions[idx, :].flatten(), self.k_candidates, target=True)

        q_pred = self.Critic.forward(states, actions)  # batch_size x 1
        q_next = self.Target_Critic.forward(next_states, next_actions)  # batch_size x 1

        # apply element-wise
        # y = q_pred.copy()

        # for i in self.batch_size:
        #     if dones[i]:
        #         y[i, :] = rewards[i]
        #     else:
        #         y[i, :] = rewards[i] + self.gamma * q_next[i]

        # y = np.expand_dims(rewards, axis=1) + self.gamma * q_next * (1-dones)  # batch_size x 1
        y = np.expand_dims(rewards, axis=1) + self.gamma * q_next * np.expand_dims(1 - dones, axis=1)  # batch_size x 1

        # print(y.shape)
        # print(rewards.shape)
        # print(q_next.shape)
        # print(dones.shape)

        super().Critic.train(states, actions, y)

        proto_actions = self.Actor.forward(states)
        action_grads = self.Critic.get_action_grads(states, proto_actions)  # actor are updated wrt to proto action
        self.Actor.train(states, action_grads)

        self.Target_Actor.soft_update(self.Actor.model)
        self.Target_Critic.soft_update(self.Critic.model)

if __name__ == '__main__':
    # # make model dirs
    # os.makedirs('./saved_models/wolp/actor', exist_ok=True)
    # os.makedirs('./saved_models/wolp/critic', exist_ok=True)

    # saved model path
    actor_path = './saved_models/wolp/actor/actor_weights.h5'
    critic_path = './saved_models/wolp/critic/critic_weights.h5'

    # initialize agent and env
    env = gym.make('SpaceInvaders-ram-v0')
    agent = wolp_agent(
            state_dim=env.observation_space.shape[0],
            action_dim=1,
            lr_a=0.0001,
            lr_c=0.001,
            buffer_size=1e5,
            batch_size=64,
            gamma=0.99,
            tau=0.001,
            save_graph=False)
    try:
        agent.restore(actor_path, critic_path)
        logging.info('restore saved model')
    except:
        pass

    # training
    rewards = []                        # list containing scores from each episode
    rewards_window = deque(maxlen=100)  # last 100 scores

    for i_episode in range(200):
        steps = 0
        ep_start_time = time.time()
        eps_reward = 0
        state = env.reset()

        while True:
            # env.render()
            # if len(agent.Buffer.memory) < agent.buffer_size:
            #     action = np.random.choice(env.action_space.n)
            # else:
            action = agent.select_action(state, 2)

            # action = agent.act(state, 2)
            next_state, reward, done, _ = env.step(action)
            if (next_state == state).all():
                action = np.random.choice(env.action_space.n)
                next_state, reward, done, _ = env.step(action)

            agent.store(state, action, reward, next_state, done)

            if len(agent.Buffer.memory) >= agent.batch_size:
                # print('agent is learning')
                agent.learn()

            state = next_state
            eps_reward += reward
            steps += 1
            if done:
                # print(steps)
                break

        rewards_window.append(eps_reward)  # save most recent score
        rewards.append(eps_reward)  # save most recent score
        print('\rEpisode {}\tAverage Score: {:.2f} | completed in {:.2f} s'.format(i_episode, np.mean(rewards_window), time.time() - ep_start_time))

        if np.mean(rewards_window) > 350:
            print('Env solved!')
            agent.save(actor_path, critic_path)
            break

        agent.save(actor_path, critic_path)
        logging.info('save model weights')
    env.close()

    # # save weights
    # agent.save(actor_path, critic_path)
    # logging.info('save model weights')