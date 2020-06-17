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

class wolp_agent:
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
                 tau,):
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

        # replay buffer
        self.Buffer = ReplayBuffer(self.buffer_size)

        # create actor and critic networks
        self.Actor = ActorNet(self.state_dim, self.action_dim, 128, 128, lr=self.lr_a, tau=self.tau)
        self.Critic = CriticNet(self.state_dim, self.action_dim, 128, 128, lr=self.lr_c, tau=self.tau)
        self.Target_Actor = ActorNet(self.state_dim, self.action_dim, 128, 128, lr=self.lr_a, tau=self.tau)
        self.Target_Critic = CriticNet(self.state_dim, self.action_dim, 128, 128, lr=self.lr_c, tau=self.tau)

        # record the learning steps
        self.learning_step = 0

    def act(self, state, k_candidates):
        proto_action = self.Actor.forward(state)
        # noised_proto_action = proto_action + OU.noise()

        action = self.refine_action(state, proto_action, k_candidates, target=False)
        try:
            action = action.flatten()
        except:
            pass
        return np.array(action)

    def refine_action(self, state, proto_action, k_candidates, target):
        if LEGAL_ACTION_CONSTRAINT:
            available_actions = [x for x in range(env.n_pipes) if state[x] > 0]  # available actions at current state
        else:
            available_actions = np.arange(env.action_space.n).tolist()

        nearest_actions = {}
        for d in range(proto_action.shape[0]):
            # find k nearest available actions in dth dimension
            nearest_actions_ids = np.argsort(np.abs(proto_action[d] - np.asarray(available_actions)))[:k_candidates]
            nearest_actions[d] = [available_actions[i] for i in nearest_actions_ids]
            # update available actions
            [available_actions.remove(x) for x in list(nearest_actions[d])]

        cand_actions = np.array(list(product(*nearest_actions.values())))  # find the power set of the candidate actions
        states = np.tile(state, [cand_actions.shape[0], 1])

        # print(cand_actions)

        if target:
            qs = self.Target_Critic.forward(states, cand_actions)  # evaluate by target critic
        else:
            qs = self.Critic.forward(states, cand_actions)

        idx_max = np.argmax(qs, axis=0)

        best_action = cand_actions[idx_max]

        return best_action.item()
        # return best_action.flatten()

    def store(self, state, action, reward, next_state, done):
        self.Buffer.store(state, action, reward, next_state, done)

    def learn(self):
        states, actions, rewards, next_states, dones = self.Buffer.sample(self.batch_size)

        # get next actions from target actor
        next_proto_actions = self.Target_Actor.forward(next_states)  # batch_size x action_dim

        # actions at next state are generated from full target policy (refined)
        next_actions = next_proto_actions.copy()
        for idx in range(next_proto_actions.shape[0]):
            next_actions[idx, :] = self.refine_action(next_states[idx, :], next_proto_actions[idx, :].flatten(), 3, target=True)

        q_pred = self.Critic.forward(states, actions)  # batch_size x 1
        q_next = self.Target_Critic.forward(next_states, next_actions)  # batch_size x 1

        # y = np.expand_dims(rewards, axis=1) + self.gamma * q_next * (1-dones)  # batch_size x 1
        y = np.expand_dims(rewards, axis=1) + self.gamma * q_next * np.expand_dims(1 - dones, axis=1)  # batch_size x 1

        # for single state
        if len(states.shape) < 2:
            states = np.expand_dims(states, axis=0)
        # for scalar actions
        if len(actions.shape) < 2:
            actions = np.expand_dims(actions, axis=1)

        self.Critic.model.train_on_batch(x=[states, actions], y=y)

        with tf.GradientTape() as tape:
            proto_actions = self.Actor.forward(states)
            actor_loss = -tf.reduce_mean(self.Critic.forward(states, proto_actions))

        actor_grad = tape.gradient(actor_loss, self.Actor.model.trainable_variables)
        self.Actor.optimizer.apply_gradients(zip(actor_grad, self.Actor.model.trainable_variables))

        # learning counter
        self.learning_step += 1

        if self.learning_step % self.a_target_update_steps == 0:
            self.Target_Actor.soft_update(self.Actor.model)
        if self.learning_step % self.c_target_update_steps == 0:
            self.Target_Critic.soft_update(self.Critic.model)

    def save(self, actor_path, critic_path):
        self.Actor.save_weights(actor_path)
        self.Critic.save_weights(critic_path)

    def restore(self, actor_path, critic_path):
        self.Actor.load_weights(actor_path)
        self.Critic.load_weights(critic_path)

############################## ACTOR ###############################

class ActorNet:
    def __init__(self, state_dim, action_dim, fc1_units, fc2_units, lr, tau):
        self.tau = tau
        # create network
        self.model = self._create_net(state_dim, action_dim, fc1_units, fc2_units)
        self.optimizer = tf.train.AdamOptimizer(lr)

    def _create_net(self, state_dim, action_dim, fc1_units, fc2_units):
        state = Input(shape=(state_dim,))
        x = Dense(fc1_units, activation='relu')(state)
        x = Dense(fc2_units, activation='relu')(x)
        x = Dense(action_dim, activation='linear')(x)
        out = ReLU(max_value=env.action_space.n)(x)   # clip the output to be within the pipe number range
        # out = tf.clip_by_value(x, 0, env.action_space.n)

        model = Model(inputs=state, outputs=out)

        return model

    def forward(self, state):
        if len(state.shape) < 2:
            state = np.expand_dims(state, axis=0)
        action = self.model.predict(state)

        return action

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
    def __init__(self, state_dim, action_dim, fc1_units, fc2_units, lr, tau):
        self.tau = tau
        # create network
        self.model = self._create_net(state_dim, action_dim, fc1_units, fc2_units)
        self.optimizer = tf.train.AdamOptimizer(lr)
        # self.loss = tf.keras.losses.mean_squared_error()
        self.model.compile(optimizer=self.optimizer,
                           loss='mse')

    def _create_net(self, state_dim, action_dim, fc1_units, fc2_units):
        state = Input(shape=(state_dim,))
        action = Input(shape=(action_dim,))
        x = Dense(fc1_units, activation='relu')(state)
        x = concatenate([x, action], axis=1)  # concat transformed state and raw action as input for fc2
        x = Dense(fc2_units, activation='relu')(x)
        out = Dense(1)(x)

        model = Model(inputs=[state, action], outputs=out)

        return model

    def forward(self, state, action):
        q = self.model.predict([state, action])

        return q

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
        self.transition = namedtuple('Transitions', ['state', 'action', 'reward', 'next_state', 'done'])

    def store(self, state, action, reward, next_state, done):
        transition = self.transition(state, action, reward, next_state, done)
        self.memory.append(transition)

    def sample(self, batch_size):
        batch = random.sample(self.memory, batch_size)
        transitions = zip(*batch)  # unfold the transition tuples and make s tuple, a tuple ...
        states, actions, rewards, next_states, dones = map(np.array, transitions)  # map the transitions to arrays

        return states, actions, rewards, next_states, dones

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
            lr_a=0.00005,
            lr_c=0.0001,
            buffer_size=1e5,
            batch_size=64,
            a_target_update_steps=1,
            c_target_update_steps=1,
            gamma=0.95,
            tau=0.01)
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

            if len(agent.Buffer.memory) == agent.buffer_size:
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

    # save weights
    agent.save(actor_path, critic_path)
    logging.info('save model weights')