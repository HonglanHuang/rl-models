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
from OU_noise import OUNoise

# hyper
LEGAL_ACTION_CONSTRAINT = False

############################## WOLP Agent #############################
# define transition named tuple
Transitions = namedtuple('Transitions', ['state', 'action', 'reward', 'next_state', 'done'])

class wolp_agent:
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

        # replay buffer
        self.Buffer = ReplayBuffer(self.buffer_size)

        # create actor and critic networks
        self.Actor = ActorNet(self.sess, self.state_dim, self.action_dim, self.action_range, 400, 300, lr=self.lr_a, tau=self.tau)
        self.Critic = CriticNet(self.sess, self.state_dim, self.action_dim, 400, 300, lr=self.lr_c, tau=self.tau)
        self.Target_Actor = ActorNet(self.sess, self.state_dim, self.action_dim, self.action_range, 400, 300, lr=self.lr_a, tau=self.tau)
        self.Target_Critic = CriticNet(self.sess, self.state_dim, self.action_dim, 400, 300, lr=self.lr_c, tau=self.tau)

        # initialize variables
        self.sess.run(tf.global_variables_initializer())
        # save graoh
        if save_graph:
            tf.summary.FileWriter('./logs/', self.sess.graph)

    def act(self, state):
        proto_action = self.Actor.forward(state)
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

    def store(self, state, action, reward, next_state, done):
        self.Buffer.store(state, action, reward, next_state, done)

    def learn(self):
        states, actions, rewards, next_states, dones = self.Buffer.sample(self.batch_size)

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

        self.Critic.train(states, actions, y)

        proto_actions = self.Actor.forward(states)
        action_grads = self.Critic.get_action_grads(states, proto_actions)  # actor are updated wrt to proto action
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
    def __init__(self, sess, state_dim, action_dim, action_range, fc1_units, fc2_units, lr, tau):
        self.sess = sess
        self.action_range = action_range
        self.tau = tau

        # create network
        self.model = self.create_net(state_dim, action_dim, fc1_units, fc2_units)

        self.optimizer = tf.train.AdamOptimizer(lr)  # optimizer, note the version

        self.actor_params = self.model.trainable_weights
        self.action_grads = tf.placeholder(tf.float32, [None, action_dim])  # receive action gradients from the critic

        self.grads = tf.gradients(self.model.output, self.actor_params, -self.action_grads)
        self.train_op = self.optimizer.apply_gradients(zip(self.grads, self.actor_params))

    def create_net(self, state_dim, action_dim, action_range, fc1_units, fc2_units):
        state = Input(shape=(state_dim,))
        x = Dense(fc1_units, kernel_initializer='he_normal')(state)
        x = BatchNormalization()(x)
        x = ReLU()(x)
        x = Dense(fc2_units, kernel_initializer='he_normal')(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)
        x = Dense(action_dim,
                  kernel_initializer=initializers.RandomUniform(-3e-3, 3e-3),
                  bias_initializer=initializers.RandomUniform(minval=-3e-4, maxval=3e-4),
                  activation='tanh')(x)
        out = Lambda(lambda x: x * action_range)(x)
        model = Model(inputs=state, outputs=out)

        return model

    def forward(self, state):
        if len(state.shape) < 2:
            state = np.expand_dims(state, axis=0)
        action = self.model.predict(state)

        try:
            action = action.flatten()
        except:
            pass

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
        # self.loss = tf.keras.losses.mean_squared_error()
        self.model.compile(optimizer=self.optimizer,
                           loss='mse')

        self.critic_params = self.model.trainable_variables
        self.action_grads = tf.gradients(self.model.output, self.model.inputs[1])  # dQ/da

    def create_net(self, state_dim, action_dim, fc1_units, fc2_units):
        state = Input(shape=(state_dim,))
        action = Input(shape=(action_dim,))
        s = Dense(fc1_units,
                  kernel_initializer='he_normal')(state)
        s = BatchNormalization()(s)  # batch norm after processing state
        s = ReLU()(s)
        a = Dense(fc1_units,
                  kernel_initializer='he_normal')(action)
        a = BatchNormalization()(a)  # batch norm after processing state
        a = ReLU()(a)
        x = concatenate([s, a], axis=1)   # concat transformed state and raw action as input for fc2
        x = Dense(fc2_units,
                  kernel_initializer='he_normal')(x)
        x = BatchNormalization()(x) # batch norm after processing state
        x = ReLU()(x)
        out = Dense(1,
                    kernel_initializer=initializers.RandomUniform(minval=-3e-3, maxval=3e-3),
                    bias_initializer=initializers.RandomUniform(minval=-3e-4, maxval=3e-4))(x)

        model = Model(inputs=[state, action], outputs=out)

        return model

    def forward(self, state, action):
        if len(state.shape) < 2:
            state = np.expand_dims(state, axis=0)
        # for scalar actions
        if len(action.shape) < 2:
            action = np.expand_dims(action, axis=1)

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

        action_grads /= state.shape[0]

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
            action = agent.act(state, 2)

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