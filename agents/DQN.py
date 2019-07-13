import tensorflow as tf
import gym
import numpy as np
from collections import deque, namedtuple
import random
import matplotlib.pyplot as plt

# model path
model_path = './saved_dqn_model/'

class DQN(object):
    def __init__(self,
                 state_dim,
                 action_dim,
                 lr,
                 buffer_size,
                 batch_size,
                 target_update_steps,
                 eps,
                 gamma,
                 tau):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.lr = lr
        self.eps = eps
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
        model = tf.keras.Sequential([
                tf.keras.layers.Dense(fc1_units, input_shape=(state_dim, ), activation=tf.nn.relu),
                tf.keras.layers.Dense(fc2_units, activation=tf.nn.relu),
                tf.keras.layers.Dense(action_dim)
                ])
        model.compile(optimizer='adam',
              loss='mse',
              metrics=['accuracy'])

        return model

    def store(self, state, action, next_state, reward, done):
        transition = self.transition(state, action, next_state, reward, done)
        self.replay_buffer.append(transition)

    def act(self, state):
        state = state[np.newaxis,:]  # adding at batch dimension for passing forward the network
        if np.random.uniform() < self.eps:
            action = random.choice(np.arange(self.action_dim))
        else:
            q_values = self.q_net.predict(state)
            action = np.argmax(q_values, axis=1)

        return np.int(action)

    def learn(self, soft_update):
        # sample from buffer
        batch = random.sample(self.replay_buffer, self.batch_size)
        transitions = zip(*batch)

        states, actions, rewards, next_states, dones = map(np.array, transitions)

        q = self.q_net.predict(next_states)  # batch_size x action_dim

        # q for next states
        q_next = self.target_net.predict(next_states)
        q_next_best = np.max(q_next, axis=1).flatten()  # (batch_size,), max of next qs

        # q targets
        q_target = q.copy()  # batch_size x action_dim
        for idx in range(self.batch_size):  # setting only the qs correponding to batch actions to target values
            q_target[idx, actions[idx]] = rewards[idx] + self.gamma * q_next_best[idx] * (1 - dones[idx])

        result = self.q_net.fit(states, q_target, epochs=1, verbose=2)

        self.loss_hist.append(result.history['loss'])
        self.learning_steps += 1

        if self.learning_steps % self.target_update_steps == 0: # update target network weights
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

    def save_weights(self, path, overwrite=False):
        self.q_net.save_weights(path, overwrite=overwrite)


if __name__ == '__main__':
    MODEL_PATH = './saved_dqn_model/'
    env = gym.make('CartPole-v0')
    agent = DQN(state_dim=env.observation_space.shape[0],
                action_dim=env.action_space.n,
                lr=0.001,
                buffer_size=1e3,
                batch_size=64,
                target_update_steps=20,
                eps=0.1,
                gamma=0.98,
                tau=0.99
                )
    for episode in range(1000):
        state = env.reset()
        while True:
            env.render()
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            agent.store(state, action, reward, next_state, done)
            if len(agent.replay_buffer) == agent.buffer_size:
                agent.learn(soft_update=False)

            state = next_state
            if done:
                break
    env.close()

    # plot
    plt.figure(figsize=(12,8))
    plt.plot(range(len(agent.loss_hist)), agent.loss_hist)
    plt.show()