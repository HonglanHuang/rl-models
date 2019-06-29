import tensorflow as tf
import gym
import numpy as np
from collections import deque, namedtuple
import random

class DQN(object):
    def __init__(self,
                 state_dim,
                 action_dim,
                 lr,
                 buffer_size,
                 target_update,
                 eps,
                 gamma
                 ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.lr = lr,
        self.eps = eps,
        self.gamma = gamma,
        self.buffer_size = buffer_size
        self.replay_buffer = deque(maxlen=self.buffer_size)
        self.transition = namedtuple('transition', ('state', 'action', 'reward', 'next_state'))
        self.q_net = build_net(self.state_dim, self.action_dim, 128, 128)
        self.target_net = build_net(self.state_dim, self.action_dim, 128, 128)

    def build_net(self, state_dim, action_dim, fc1, fc2):
        model = tf.keras.Sequential([
                tf.keras.layers.Dense(fc1_units, input_shape=(state_dim,), activation=tf.nn.relu),
                tf.keras.layers.Dense(fc2_units, activation=tf.nn.relu)
                tf.keras.layers.Dense(action_dim)
                ])
        model.compile(optimizer='adam',
              loss='mse',
              metrics=['accuracy']))

        return model

    def store(self, transitions):
        self.replay_buffer.append(transitions)

    def act(self):
        if np.random.uniform() < self.eps:
            action = random.choice()
        else:
            action = np.argmax()
        return action

    def learn(self):
        # sample from buffer
        q_next = self.target_net.predict(s_next)
        q_target = r + gamma * q_next
        self.q_net.fit(q_target, epoch=1)
        if target_update_step:
            self.target_net.set_weights(self.q_net.get_weights())

    def load_weights(self, path):
        self.q_net.load_weights(path)
        self.target_net.set_weights(self.q_net.get_weights())

    def save_weights(self, path, overwrite=False):
        self.q_net.save_weights(path, overwrite=overwrite)

    def update_target_model_hard(self):
        self.target_net.set_weights(self.model.get_weights())
