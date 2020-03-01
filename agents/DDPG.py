import os
import gym
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.keras import Sequential, Model
from tensorflow.keras.layers import Dense, Input, ReLU, LeakyReLU, Activation, concatenate, Lambda, BatchNormalization
from tensorflow.keras import regularizers
from tensorflow.keras import initializers

from collections import deque, namedtuple
import random
import warnings
import time

# OU Noise
from OU_noise import OUNoise

############################## ACTOR ###############################

class ActorNet:
    def __init__(self, sess, state_dim, action_dim, fc1_units, fc2_units, lr, tau):
        self.sess = sess
        self.tau = tau

        # create network
        self.model = self.create_net(state_dim, action_dim, fc1_units, fc2_units)

        self.optimizer = tf.train.AdamOptimizer(lr)  # optimizer, note the version

        self.actor_params = self.model.trainable_weights
        self.action_grads = tf.placeholder(tf.float32, [None, action_dim])  # receive action gradients from the critic

        self.grads = tf.gradients(self.model.output, self.actor_params, -self.action_grads)
        self.train_op = self.optimizer.apply_gradients(zip(self.grads, self.actor_params))

    def create_net(self, state_dim, action_dim, fc1_units, fc2_units):
        state = Input(shape=(state_dim,))
        # x = Dense(fc1_units, activation='relu')(state)
        # x = Dense(fc2_units, activation='relu')(x)
        x = Dense(fc1_units, kernel_initializer='he_normal')(state)
        x = BatchNormalization()(x)
        x = LeakyReLU()(x)
        x = Dense(fc2_units, kernel_initializer='he_normal')(x)
        x = BatchNormalization()(x)
        x = LeakyReLU()(x)
        x = Dense(action_dim,
                  kernel_initializer=initializers.RandomUniform(-3e-4, 3e-4),
                  bias_initializer=initializers.RandomUniform(minval=-3e-4, maxval=3e-4),
                  activation='tanh')(x)
        # x = Dense(action_dim, activation='tanh')(x)
        out = Lambda(lambda x: x * 2)(x)

        # out = Dense(action_dim, kernel_initializer='he_normal', activation='relu')(x)
        model = Model(inputs=state, outputs=out)

        return model

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
        self.sess.run(self.train_op, feed_dict={
            self.model.input: states,                 # pass states to model input layer
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
        self.model = self.create_net(state_dim, action_dim, fc1_units, fc2_units)
        self.optimizer = tf.train.AdamOptimizer(lr)
        self.loss = tf.keras.losses.MeanSquaredError()
        self.model.compile(optimizer=self.optimizer,
                           loss=self.loss)

        self.critic_params = self.model.trainable_variables
        self.action_grads = tf.gradients(self.model.output, self.model.inputs[1])  # dQ/da

    def create_net(self, state_dim, action_dim, fc1_units, fc2_units):
        state = Input(shape=(state_dim,))
        action = Input(shape=(action_dim,))
        x = Dense(fc1_units,
                  kernel_initializer='he_normal',
                  kernel_regularizer=regularizers.l2(0.01))(state)
        x = BatchNormalization()(x)  # batch norm after processing state
        x = ReLU()(x)
        a = Dense(fc1_units,
                  kernel_initializer='he_normal',
                  kernel_regularizer=regularizers.l2(0.01))(action)
        a = BatchNormalization()(a)  # batch norm after processing state
        a = ReLU()(a)
        x = concatenate([x, a], axis=1)   # concat transformed state and raw action as input for fc2
        x = Dense(fc2_units,
                  kernel_initializer='he_normal',
                  kernel_regularizer=regularizers.l2(0.01))(x)
        x = BatchNormalization()(x) # batch norm after processing state
        x = ReLU()(x)
        out = Dense(1,
                    kernel_initializer=initializers.RandomUniform(minval=-3e-3, maxval=3e-3),
                    bias_initializer=initializers.RandomUniform(minval=-3e-3, maxval=3e-3),
                    kernel_regularizer=regularizers.l2(0.01))(x)

        model = Model(inputs=[state, action], outputs=out)

        return model

    def forward(self, state, action):
        q = self.model.predict([state, action])

        return q

    def train(self, states, actions, y):
        self.model.train_on_batch(x=[states, actions], y=y)

    def get_action_grads(self, states, actions):
        if len(states.shape) < 2:
            states = np.expand_dims(states, axis=0)
        # for scalar actions
        if len(actions.shape) < 2:
            actions = np.expand_dims(actions, axis=1)

        action_grads = self.sess.run(self.action_grads, feed_dict={
            self.model.inputs[0]: states,
            self.model.inputs[1]: actions
        })[0]

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
        self.Buffer = deque(maxlen=buffer_size)
        self.transition_tuple = namedtuple('Transitions', ['s', 'a', 'r', 's_n', 'done'])

    def store(self, state, action, reward, next_state, done):
        transition = self.transition_tuple(state, action, reward, next_state, done)
        self.Buffer.append(transition)

    def sample(self, batch_size):
        batch = random.sample(self.Buffer, batch_size)
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

############################## DDPG Agent #############################

class DDPG:
    def __init__(self,
                 state_dim,
                 action_dim,
                 lr_a,
                 lr_c,
                 buffer_size,
                 batch_size,
                 a_target_update_steps,
                 c_target_update_steps,
                 gamma,
                 tau,
                 save_graph):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.lr_a = lr_a
        self.lr_c = lr_c
        self.gamma = gamma
        self.buffer_size = int(buffer_size)
        self.batch_size = int(batch_size)
        self.tau = tau

        self.a_target_update_steps = a_target_update_steps
        self.c_target_update_steps = c_target_update_steps

        # tf session
        # config = tf.compat.v1.ConfigProto()
        # config.gpu_options.allow_growth = True
        self.sess = tf.Session()

        # replay buffer
        self.Buffer = ReplayBuffer(self.buffer_size)

        # create actor and critic networks
        self.Actor = ActorNet(self.sess, self.state_dim, self.action_dim, 256, 128, lr=self.lr_a, tau=self.tau)
        self.Critic = CriticNet(self.sess, self.state_dim, self.action_dim, 256, 128, lr=self.lr_c, tau=self.tau)
        self.Target_Actor = ActorNet(self.sess, self.state_dim, self.action_dim, 256, 128, lr=self.lr_a, tau=self.tau)
        self.Target_Critic = CriticNet(self.sess, self.state_dim, self.action_dim, 256, 128, lr=self.lr_c, tau=self.tau)

        # initialize variables
        self.sess.run(tf.global_variables_initializer())

        if save_graph:
            tf.summary.FileWriter('./logs/', self.sess.graph)

        # timer
        self.learning_step = 0

    def act(self, state):
        raw_action = self.Actor.forward(state)
        noised_action = raw_action + OU.noise()

        return noised_action.flatten()

    def store(self, state, action, reward, next_state, done):
        self.Buffer.store(state, action, reward, next_state, done)

    def learn(self):
        # note, action batch here is only used for updating critic
        states, actions, rewards, next_states, dones = self.Buffer.sample(self.batch_size)

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

        # learning counter
        self.learning_step += 1

        if self.learning_step % self.a_target_update_steps == 0:
            self.Target_Actor.soft_update(self.Actor.model)
        if self.learning_step % self.c_target_update_steps == 0:
            self.Target_Critic.soft_update(self.Critic.model)


if __name__ == '__main__':
    os.makedirs('./saved_models/ddpg/actor', exist_ok=True)
    os.makedirs('./saved_models/ddpg/critic', exist_ok=True)

    OU = OUNoise(action_dimension=1, mu=0, theta=0.15, sigma=0.2)

    env = gym.make('MountainCarContinuous-v0')
    # env = gym.make('BipedalWalker-v2')
    agent = DDPG(state_dim=env.observation_space.shape[0],
                 action_dim=env.action_space.shape[0],
                 lr_a=0.00001,
                 lr_c=0.001,
                 buffer_size=1e5,
                 batch_size=128,
                 a_target_update_steps=1,
                 c_target_update_steps=1,
                 gamma=0.99,
                 tau=0.001,
                 save_graph=False)
    try:
        agent.Actor.load_weights('./saved_models/ddpg/actor/actor_weights.h5')
        agent.Critic.load_weights('./saved_models/ddpg/critic/critic_weights.h5')
    except:
        pass

    #  run training
    rewards = []                        # list containing scores from each episode
    rewards_window = deque(maxlen=100)  # last 100 scores

    for i_episode in range(2000):
        ep_start_time = time.time()
        eps_reward = 0
        state = env.reset()
        while True:
            # env.render()
            action = agent.act(state)
            # print(action)
            next_state, reward, done, _ = env.step(action)
            agent.store(state, action, reward, next_state, done)
            if len(agent.Buffer.Buffer) == agent.buffer_size:
                agent.learn()

            state = next_state
            eps_reward += reward

            if done:
                break

        rewards_window.append(eps_reward)  # save most recent score
        rewards.append(eps_reward)  # save most recent score
        print('\rEpisode {}\tAverage Score: {:.2f} | completed in {:.2f} s'.format(i_episode, np.mean(rewards_window), time.time() - ep_start_time))

    env.close()

    # # plot the rewards
    # fig = plt.figure(figsize=(10, 8))
    # ax = fig.add_subplot(111)
    # plt.plot(np.arange(len(rewards)), rewards)
    # plt.ylabel('Rewards', fontsize=12)
    # plt.xlabel('Episode #', fontsize=12)
    # plt.show()

    # save model
    agent.Actor.save_weights('./saved_models/ddpg/actor/actor_weights.h5', overwrite=True)
    agent.Critic.save_weights('./saved_models/ddpg/critic/critic_weights.h5', overwrite=True)