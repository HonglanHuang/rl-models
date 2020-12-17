import gym
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import Sequential, Model
from tensorflow.keras.layers import Dense, Input, ReLU

from collections import deque

import os
import time
import random
import warnings

############################## ACTOR ###############################

class ActorNet:
    def __init__(self, state_dim, action_dim, fc1_units, fc2_units, lr):
        # create network
        self.model = self.create_net(state_dim, action_dim, fc1_units, fc2_units)
        self.optimizer = tf.train.AdamOptimizer(lr)
        self.loss = tf.keras.losses.CategoricalCrossentropy()
        self.model.compile(optimizer=self.optimizer,
                           loss=self.loss)

    def create_net(self, state_dim, action_dim, fc1_units, fc2_units):
        state = Input(shape=(state_dim,))
        x = Dense(fc1_units, activation='relu')(state)
        x = Dense(fc2_units, activation='relu')(x)
        out = Dense(action_dim, activation='softmax')(x)

        model = Model(inputs=state, outputs=out)

        return model

    def forward(self, state):
        if len(state.shape) < 2:
            state = np.expand_dims(state, axis=0)
        action_probs = self.model.predict_on_batch(state)
        action = np.random.choice(np.arange(env.action_space.n), p=action_probs.ravel())

        return action.item()

    def train(self, state, action, advantage):
        if len(state.shape) < 2:
            state = np.expand_dims(state, axis=0)
        action_one_hot = tf.one_hot(action, depth=1)
        advantage_weight = action_one_hot * advantage
        self.model.train_on_batch(x=state, y=action_one_hot, class_weight=-tf.reshape(advantage_weight, (-1,)))

    def load_weights(self, model_path):
        self.model.load_weights(model_path)

    def save_weights(self, model_path, overwrite=True):
        self.model.save_weights(model_path, overwrite=overwrite)

############################## CRITIC ###############################

class CriticNet:
    # critic network model
    def __init__(self, state_dim, action_dim, fc1_units, fc2_units, lr):
        # create network
        self.model = self.create_net(state_dim, action_dim, fc1_units, fc2_units)
        self.optimizer = tf.train.AdamOptimizer(lr)
        self.loss = tf.keras.losses.MeanSquaredError()
        self.model.compile(optimizer=self.optimizer,
                           loss=self.loss)

    def create_net(self, state_dim, action_dim, fc1_units, fc2_units):
        state = Input(shape=(state_dim,))
        x = Dense(fc1_units, activation='relu')(state)
        x = Dense(fc2_units, activation='relu')(x)
        out = Dense(1)(x)

        model = Model(inputs=state, outputs=out)

        return model

    def forward(self, state):
        if len(state.shape) < 2:
            state = np.expand_dims(state, axis=0)
        q = self.model.predict_on_batch(state)

        return q

    def train(self, state, y):
        # self.optimizer.minimize(loss, self.model.trainable_weights)
        if len(state.shape) < 2:
            state = np.expand_dims(state, axis=0)
        self.model.train_on_batch(x=state, y=y)

    def load_weights(self, model_path):
        self.model.load_weights(model_path)

    def save_weights(self, model_path, overwrite=True):
        self.model.save_weights(model_path, overwrite=overwrite)

############################## A2C Agent #############################

class AC:
    def __init__(self,
                 state_dim,
                 action_dim,
                 lr_a,
                 lr_c,
                 buffer_size,
                 batch_size,
                 a_target_update_steps,
                 c_target_update_steps,
                 epsilon,
                 gamma,
                 tau):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.lr_a = lr_a
        self.lr_c = lr_c
        self.eps = epsilon
        self.gamma = gamma
        self.buffer_size = int(buffer_size)
        self.batch_size = int(batch_size)
        self.tau = tau

        # create actor and critic networks
        self.Actor = ActorNet(self.state_dim, self.action_dim, 128, 128, lr=self.lr_a)
        self.Critic = CriticNet(self.state_dim, self.action_dim, 128, 128, lr=self.lr_c)

        # timer
        self.learning_step = 0

    def act(self, state):
        action = self.Actor.forward(state)

        return action

    def learn(self, state, action, reward, next_state, done):
        state, next_state = np.expand_dims(state, axis=0), np.expand_dims(next_state, axis=0)

        # critic
        v_pred = self.Critic.forward(state)  # batch_size x 1
        v_next = self.Critic.forward(next_state)  # batch_size x 1
        y = np.expand_dims(reward, axis=0) + self.gamma * v_next * (1-np.expand_dims(done, axis=1))  # batch_size x 1
        advantage = y - v_pred  # batch_size x 1
        self.Critic.train(state, y)

        # actor
        self.Actor.train(state, action, advantage)

if __name__ == '__main__':
    os.makedirs('./saved_models/a2c/actor', exist_ok=True)
    os.makedirs('./saved_models/a2c/critic', exist_ok=True)
    env = gym.make('CartPole-v0')
    agent = AC(state_dim=env.observation_space.shape[0],
                 action_dim=2,
                 lr_a=0.01,
                 lr_c=0.001,
                 buffer_size=1e3,
                 batch_size=64,
                 a_target_update_steps=10,
                 c_target_update_steps=10,
                 epsilon=0.1,
                 gamma=0.9,
                 tau=0.95)
    try:
        agent.restore()
    except:
        pass

    #  run training
    rewards = []                        # list containing scores from each episode
    rewards_window = deque(maxlen=100)  # last 100 scores
    for i_episode in range(100):
        ep_start_time = time.time()
        eps_reward = 0
        state = env.reset()
        while True:
            env.render()
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            agent.learn(state, action, reward, next_state, done)

            # state transition
            state = next_state
            eps_reward += reward
            if done:
                break

        rewards_window.append(eps_reward)  # save most recent score
        rewards.append(eps_reward)  # save most recent score
        print('\rEpisode {}\tAverage Score: {:.2f} | completed in {:.2f} s'.format(i_episode, np.mean(rewards_window),
                                                                                   time.time() - ep_start_time))
    env.close()

    # plot the rewards
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111)
    plt.plot(np.arange(len(rewards)), rewards)
    plt.ylabel('Rewards', fontsize=12)
    plt.xlabel('Episode #', fontsize=12)
    plt.show()

    # save model
    agent.Actor.save_weights('./saved_models/a2c/actor/actor_weights.h5', overwrite=True)
    agent.Critic.save_weights('./saved_models/a2c/critic/critic_weights.h5', overwrite=True)