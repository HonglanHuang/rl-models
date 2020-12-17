import tensorflow as tf
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Dense, ReLU, Input, concatenate
from tensorflow.keras.optimizers import Adam
import gym
import numpy as np
from collections import deque, namedtuple
import random
import matplotlib.pyplot as plt

# model path
model_path = './saved_dqn_model/'

# Hyperparameters
LR = 0.001
BUFFER_SIZE = 1e4
BATCH_SIZE = 128
EPSILON = 0.95
EPS_DECAY = 0.99
EPS_MIN = 0.1
GAMMA = 0.95
TARGET_UPDATE_STEP = 100
TAU = 0.95

class DQN(object):
    # Deep Q Network model
    def __init__(self,
                 state_dim,
                 action_dim,
                 lr,
                 buffer_size,
                 batch_size,
                 target_update_steps,
                 eps,
                 eps_decay,
                 eps_min,
                 gamma,
                 tau):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.lr = lr
        self.eps = eps
        self.eps_decay = eps_decay
        self.eps_min = eps_min
        self.gamma = gamma
        self.buffer_size = int(buffer_size)
        self.batch_size = int(batch_size)
        self.replay_buffer = deque(maxlen=self.buffer_size)
        self.transition = namedtuple('transition', ('state', 'action', 'reward', 'next_state', 'done'))
        self.q_net = self.build_net(self.state_dim, self.action_dim, 128, 128)
        self.target_net = self.build_net(self.state_dim, self.action_dim, 128, 128)
        self.learning_steps = 0
        self.target_update_steps = target_update_steps
        self.loss_hist = []

    def build_net(self, state_dim, action_dim, fc1_units, fc2_units):
        model = Sequential([
                Dense(fc1_units, input_shape=(state_dim, ), activation='relu'),
                Dense(fc2_units, activation='relu'),
                Dense(action_dim)
                ])
        model.compile(optimizer=Adam(lr=0.01),
              loss='mse')

        return model

    def store(self, state, action, next_state, reward, done):
        transition = self.transition(state, action, next_state, reward, done)
        self.replay_buffer.append(transition)

    def act(self, state):
        if len(state.shape) < 2:
            state = state[np.newaxis,:]  # adding at batch dimension for passing forward the network
        # epsilon greedy
        if np.random.uniform() < self.eps:
            action = random.choice(np.arange(self.action_dim))
        else:
            q_values = self.q_net.predict(state)
            action = np.argmax(q_values, axis=1)

        return np.int(action)

    def learn(self, soft_update):
        # sample from buffer
        batch = random.sample(self.replay_buffer, self.batch_size)
        transitions = zip(*batch)  # make (st, at, st+1, rt) batches to (si), (ai), (st+1), (ri) tuples

        states, actions, rewards, next_states, dones = map(np.array, transitions)

        q = self.q_net.predict(next_states)  # batch_size x action_dim

        # q for next states
        q_next = self.target_net.predict(next_states)
        q_next_best = np.max(q_next, axis=1).flatten()  # (batch_size,), max of next qs

        # q targets
        q_target = q.copy()  # batch_size x action_dim
        for idx in range(self.batch_size):  # setting only the qs correponding to batch actions to target values
            q_target[idx, actions[idx]] = rewards[idx] + self.gamma * q_next_best[idx] * (1 - dones[idx])

        result = self.q_net.train_on_batch(states, q_target)

        self.loss_hist.append(result)
        self.learning_steps += 1
        if self.eps > self.eps_min:
            self.eps *= self.eps_decay

        if self.learning_steps % self.target_update_steps == 0:  # update target network weights
            q_w = self.q_net.get_weights()
            target_w = self.target_net.get_weights()
            if soft_update:
                for idx in range(len(target_w)):
                    target_w[idx] = self.tau * q_w[idx] + (1-self.tau) * target_w[idx]
                self.target_net.set_weights(target_w)
            else:
                self.target_net.set_weights(q_w)


    def load_weights(self, path):
        self.q_net.load_weights(path)
        self.target_net.set_weights(self.q_net.get_weights())

    def save_weights(self, path, overwrite=True):
        self.q_net.save_weights(path, overwrite=overwrite)


if __name__ == '__main__':
    MODEL_PATH = './saved_dqn_model/'
    env = gym.make('CartPole-v1')
    agent = DQN(state_dim=env.observation_space.shape[0],
                action_dim=env.action_space.n,
                lr=LR,
                buffer_size=BUFFER_SIZE,
                batch_size=BATCH_SIZE,
                target_update_steps=TARGET_UPDATE_STEP,
                eps=EPSILON,
                eps_decay=EPS_DECAY,
                eps_min=EPS_MIN,
                gamma=GAMMA,
                tau=TAU)


    # if TEST:
    #     agent.restore()  # restore the trained agent

    for episode in range(1000):
        state = env.reset()
        for iter in range(500):
            env.render()
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            agent.store(state, action, reward, next_state, done)
            if len(agent.replay_buffer) == agent.buffer_size:
                agent.learn(soft_update=False)

            state = next_state
            if done:
                print("episode: {}/{} completed, total reward: {}, epsilon: {:.2}"
                      .format(episode, 1000, iter, agent.eps))
                break

    env.close()

    # plot
    plt.plot(range(len(agent.loss_hist)), agent.loss_hist)
    plt.show()