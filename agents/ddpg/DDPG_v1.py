import os
import gym
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.keras import Sequential, Model
from tensorflow.keras.layers import Dense, Input, ReLU, LeakyReLU, Activation, concatenate, Lambda, BatchNormalization, Multiply
from tensorflow.keras import regularizers
from tensorflow.keras import initializers

import tensorflow.keras.backend as K

from collections import deque, namedtuple
import random
import warnings
import time

# OU Noise
# from OU_noise import OUNoise

############################## DDPG Agent #############################
# define transition named tuple
Transitions = namedtuple('Transitions', ['state', 'action', 'reward', 'next_state', 'done'])

class DDPG:
    def __init__(self,
                 state_dim,
                 action_dim,
                 action_range,
                 lr_a,
                 lr_c,
                 buffer_size,
                 batch_size,
                 gamma,
                 tau,
                 save_graph):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_range = action_range
        self.lr_a = lr_a
        self.lr_c = lr_c
        self.gamma = gamma
        self.buffer_size = int(buffer_size)
        self.batch_size = int(batch_size)
        self.tau = tau

        # tf session
        # config = tf.compat.v1.ConfigProto()
        # config.gpu_options.allow_growth = True
        self.sess = tf.Session()

        # replay buffer
        self.Buffer = ReplayBuffer(self.buffer_size)

        # create actor and critic networks
        self.Actor = ActorNet(self.sess, self.state_dim, self.action_dim, self.action_range, 64, 32, self.lr_a, self.tau, self.batch_size)
        self.Critic = CriticNet(self.sess, self.state_dim, self.action_dim, 64, 32, self.lr_c, self.tau)
        self.Target_Actor = ActorNet(self.sess, self.state_dim, self.action_dim, self.action_range, 64, 32, self.lr_a, self.tau, self.batch_size)
        self.Target_Critic = CriticNet(self.sess, self.state_dim, self.action_dim, 64, 32, self.lr_c, self.tau)

        # initialize variables
        self.sess.run(tf.global_variables_initializer())

        if save_graph:
            tf.summary.FileWriter('./logs/', self.sess.graph)

    def act(self, state):
        raw_action = self.Actor.forward(state)
        # print(raw_action)
        noised_action = raw_action + OU()
        noised_action = np.clip(noised_action.flatten(), env.action_space.low, env.action_space.high)
        # print(noised_action)

        return noised_action
        # return raw_action.flatten()

    def store(self, state, action, reward, next_state, done):
        self.Buffer.store(state, action, reward, next_state, done)

    def learn(self):
        # note, action batch here is only used for updating critic
        states, actions, rewards, next_states, dones = self.Buffer.sample(self.batch_size)
        # print(rewards)
        # print(dones.shape)

        # compute q target
        next_actions = self.Target_Actor.forward(next_states)  # batch_size x action_dim
        q_pred = self.Critic.forward(states, actions)  # batch_size x 1
        q_next = self.Target_Critic.forward(next_states, next_actions)  # batch_size x 1
        y = np.expand_dims(rewards, axis=1) + self.gamma * q_next * (1-np.expand_dims(dones, axis=1))  # batch_size x 1

        # train critic
        self.Critic.train(states, actions, y)

        # train actor
        action_preds = self.Actor.forward(states)  # batch x action_dim  need prediction because the gradient is wrt policy params
        action_grads = self.Critic.get_action_grads(states, action_preds)
        self.Actor.train(states, action_grads)

        self.Target_Actor.soft_update(self.Actor.model)
        self.Target_Critic.soft_update(self.Critic.model)

    def save(self, actor_path, critic_path):
        self.Actor.save_weights(actor_path)
        self.Critic.save_weights(critic_path)

    def restore(self, actor_path, critic_path):
        self.Actor.load_weights(actor_path)
        self.Critic.load_weights(critic_path)

############################## ACTOR ###############################

class ActorNet:
    def __init__(self, sess, state_dim, action_dim, action_range, fc1_units, fc2_units, lr, tau, batch_size):
        self.sess = sess
        self.action_range = action_range
        self.tau = tau
        self.batch_size = batch_size

        # create network
        # self.model = self.create_net(state_dim, action_dim, action_range, fc1_units, fc2_units)
        self.model, self.s_in = self.create_net(state_dim, action_dim, action_range, fc1_units, fc2_units)

        self.optimizer = tf.train.AdamOptimizer(lr)  # optimizer, note the version

        self.actor_params = self.model.trainable_weights
        self.action_grads = tf.placeholder(tf.float32, [None, action_dim])  # receive action gradients from the critic

        self.grads = tf.gradients(self.model.outputs, self.actor_params, -self.action_grads)
        # self.normalized_grads = list(map(lambda x: tf.div(x, self.batch_size), self.grads))

        self.train_op = self.optimizer.apply_gradients(zip(self.grads, self.actor_params))
        # self.train_op = self.optimizer.apply_gradients(zip(self.normalized_grads, self.actor_params))

    def create_net(self, state_dim, action_dim, action_range, fc1_units, fc2_units):
        S = Input(shape=(state_dim,))
        x = Dense(fc1_units)(S)
        # x = BatchNormalization()(x)
        x = ReLU()(x)
        x = Dense(fc2_units)(x)
        # x = BatchNormalization()(x)
        x = ReLU()(x)
        x = Dense(action_dim,
                  # kernel_initializer=initializers.RandomUniform(-3e-3, 3e-3),
                  # kernel_initializer=initializers.VarianceScaling(),
                  # bias_initializer=initializers.VarianceScaling(),
                  activation='tanh')(x)

        out = Lambda(lambda x: x * action_range)(x)
        # out = action_range * out

        model = Model(inputs=S, outputs=out)

        return model, S

    def forward(self, state):
        if len(state.shape) < 2:
            state = np.expand_dims(state, axis=0)

        action = self.model.predict(state)

        return action

    def train(self, states, action_grads):
        """
        apply dQ/d_theta = dQ/da * da/d_theta
        put dQ/da at grad_ys
        make -dQ/da in order to ascend the policy towards Q
        """
        # self.sess.run(self.train_op, feed_dict={
        #     self.model.input: states,                 # pass states to model input layer
        #     self.action_grads: action_grads
        # })

        self.sess.run(self.train_op, feed_dict={
            self.s_in: states,                      # pass states to model input layer
            self.action_grads: action_grads
        })

        # # non sess method
        # self.model.train_on_batch(x=state, y=action, sample_weight=-action_grads)  # resulting loss = dQ/da * da/a_theta

    def load_weights(self, model_path):
        self.model.load_weights(model_path)

    def save_weights(self, model_path, overwrite=True):
        self.model.save_weights(model_path, overwrite=overwrite)

    def soft_update(self, actor_net):
        actor_w = actor_net.get_weights()
        model_w = self.model.get_weights()
        for idx in range(len(model_w)):
            model_w[idx] = self.tau * actor_w[idx] + (1 - self.tau) * model_w[idx]
        self.model.set_weights(model_w)

############################## CRITIC ###############################

class CriticNet:
    # critic network model
    def __init__(self, sess, state_dim, action_dim, fc1_units, fc2_units, lr, tau):
        self.sess = sess
        self.tau = tau
        # create network
        self.model, self.s_in, self.a_in, self.out = self.create_net(state_dim, action_dim, fc1_units, fc2_units)

        # self.model = self.create_net(state_dim, action_dim, fc1_units, fc2_units)
        self.optimizer = tf.train.AdamOptimizer(lr)
        # self.loss = tf.keras.losses.mean_squared_error()
        # self.model.compile(optimizer=self.optimizer,
        #                    loss='mse')

        self.critic_params = self.model.trainable_variables
        # self.action_grads = tf.gradients(self.model.output, self.model.inputs[1])  # dQ/da
        # self.action_grads = tf.gradients(self.model.outputs, self.a_in)  # dQ/da

        self.action_grads = K.function([self.model.input[0], self.model.input[1]],
                                       K.gradients(self.model.output, [self.model.input[1]]))

        # Network target (y_i)
        self.y = tf.placeholder(tf.float32, [None, 1])

        # Define loss and optimization Op
        self.loss = tf.math.reduce_mean(tf.math.square(self.y - self.out))
        self.train_op = self.optimizer.minimize(self.loss)

    def create_net(self, state_dim, action_dim, fc1_units, fc2_units):
        S = Input(shape=(state_dim,))
        A = Input(shape=(action_dim,))
        s = Dense(fc1_units)(S)
        # s = BatchNormalization()(s)  # batch norm after processing state
        s = ReLU()(s)
        s = Dense(fc2_units)(s)
        a = Dense(fc2_units)(A)
        x = concatenate([s, a], axis=1)   # concat transformed state and raw action as input for fc2
        x = ReLU()(x)
        # x = Dense(fc2_units, activation='relu')(x)
        # x = BatchNormalization()(x) # batch norm after processing state
        # x = ReLU()(x)
        out = Dense(1
                    # kernel_initializer=initializers.RandomUniform(minval=-3e-3, maxval=3e-3)
                    )(x)

        model = Model(inputs=[S, A], outputs=out)

        return model, S, A, out

    def forward(self, state, action):
        if len(state.shape) < 2:
            state = np.expand_dims(state, axis=0)

        # for scalar actions
        if len(action.shape) < 2:
            action = np.expand_dims(action, axis=1)

        q = self.model.predict([state, action])

        return q

    def train(self, states, actions, y):
        self.sess.run([self.out, self.train_op], feed_dict={
            self.s_in: states,
            self.a_in: actions,
            self.y: y
        })

        # self.model.train_on_batch(x=[states, actions], y=y)
        # self.model.fit(x=[states, actions], y=y, batch_size=y.shape[0]);

    def get_action_grads(self, states, actions):
        if len(states.shape) < 2:
            states = np.expand_dims(states, axis=0)
        # for scalar actions
        if len(actions.shape) < 2:
            actions = np.expand_dims(actions, axis=1)

        # action_grads = self.sess.run(self.action_grads, feed_dict={
        #     self.model.inputs[0]: states,
        #     self.model.inputs[1]: actions
        # })[0]

        # action_grads = self.sess.run(self.action_grads, feed_dict={
        #     self.s_in: states,
        #     self.a_in: actions
        # })[0]

        # action_grads /= states.shape[0]
        action_grads = self.action_grads([states, actions])[0]

        return action_grads

    def load_weights(self, model_path):
        self.model.load_weights(model_path)

    def save_weights(self, model_path, overwrite=True):
        self.model.save_weights(model_path, overwrite=overwrite)

    def soft_update(self, critic_net):
        critic_w = critic_net.get_weights()
        model_w = self.model.get_weights()
        for idx in range(len(model_w)):
            model_w[idx] = self.tau * critic_w[idx] + (1 - self.tau) * model_w[idx]
        self.model.set_weights(model_w)

############################## REPLAY BUFFER #############################

class ReplayBuffer:
    # replay buffer
    def __init__(self, buffer_size):
        self.memory = deque(maxlen=buffer_size)
        # self.transition = namedtuple('Transitions', ['s', 'a', 'r', 's_n', 'done'])

    def store(self, state, action, reward, next_state, done):
        # transition = self.transition(state, action, reward, next_state, done)
        transition = Transitions(state, action, reward, next_state, done)
        self.memory.append(transition)

    def sample(self, batch_size):
        batch = random.sample(self.memory, batch_size)
        transitions = zip(*batch)  # unfold the transition tuples and make s, a, r, s_n, done tuples grouped by type

        """
        map the transitions to arrays
        resulting dim: states batch x state_dim
                       actions batch x action_dim 
                       rewards (batch, )
                       next_states batch x state_dim
                       dones (batch, ) 
        """
        states, actions, rewards, next_states, dones = map(np.array, transitions)

        return states, actions, rewards, next_states, dones

    def save_memory(self, buffer_path):
        if buffer_path is not None:
            f = open(buffer_path, 'wb')
            pickle.dump(self.memory, f)
            f.close()

    def load_memory(self, buffer_path):
        if buffer_path is not None:
            f = open(buffer_path, 'rb')
            self.memory = pickle.load(f)
            f.close()

############################## OU Noise #############################

class OrnsteinUhlenbeckActionNoise:
    def __init__(self, mu, sigma=0.3, theta=.15, dt=1e-2, x0=None):
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.x0 = x0
        self.reset()

    def __call__(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + \
                self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        self.x_prev = x
        return x

    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)

    def __repr__(self):
        return 'OrnsteinUhlenbeckActionNoise(mu={}, sigma={})'.format(self.mu, self.sigma)

if __name__ == '__main__':
    env = gym.make('Pendulum-v0')
    # os.makedirs('./saved_models/ddpg/actor', exist_ok=True)
    # os.makedirs('./saved_models/ddpg/critic', exist_ok=True)
    ACTOR_PATH = './saved_models/ddpg/actor/actor_weights.h5'
    CRITIC_PATH = './saved_models/ddpg/critic/critic_weights.h5'
    # REPLAY_START = 1e2

    OU = OrnsteinUhlenbeckActionNoise(mu=np.zeros(env.action_space.shape[0]))
    agent = DDPG(state_dim=env.observation_space.shape[0],
                 action_dim=env.action_space.shape[0],
                 action_range=env.action_space.high,
                 lr_a=0.001,
                 lr_c=0.002,
                 buffer_size=1e4,
                 batch_size=32,
                 gamma=0.9,
                 tau=0.01,
                 save_graph=False)
    try:
        agent.restore(ACTOR_PATH, CRITIC_PATH)
        print('restored agent')
    except:
        pass

    #  run training
    rewards = []                        # list containing scores from each episode
    rewards_window = deque(maxlen=100)  # last 100 scores

    for i_episode in range(1000):
        ep_start_time = time.time()
        eps_reward = 0
        state = env.reset()
        while True:
            # env.render()
            action = agent.act(state)
            # print(action)
            next_state, reward, done, _ = env.step(action)
            agent.store(state, action, reward, next_state, done)
            if len(agent.Buffer.memory) >= agent.batch_size:
                # print('agent is learning')
                agent.learn()

            state = next_state
            eps_reward += reward

            if done:
                break

        rewards_window.append(eps_reward)  # save most recent score
        rewards.append(eps_reward)  # save most recent score
        print('\rEpisode {}\tAverage Score: {:.2f} | completed in {:.2f} s'.format(i_episode, np.mean(rewards_window), time.time() - ep_start_time))

        if np.mean(rewards_window) > -150:
            print('Env solved!')
            agent.save(actor_path, critic_path)
            break

    env.close()

    # plot the rewards
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111)
    plt.plot(np.arange(len(rewards)), rewards)
    plt.ylabel('Rewards', fontsize=12)
    plt.xlabel('Episode #', fontsize=12)
    plt.show()

    # save model
    # agent.Actor.save_weights('./saved_models/ddpg/actor/actor_weights.h5', overwrite=True)
    # agent.Critic.save_weights('./saved_models/ddpg/critic/critic_weights.h5', overwrite=True)