import threading
import gym
import tensorflow as tf
import model
import numpy as np
import os
from datetime import datetime


class History:
    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []

    def append(self, state, action, reward):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)

    def clear(self):
        self.states = []
        self.actions = []
        self.rewards = []


class Worker(threading.Thread):
    global_episode = 0
    global_average_running_reward = 0
    best_score = 0
    save_lock = threading.Lock()

    def __init__(self,
                 worker_index,
                 global_network,
                 gym_game_name,
                 max_episodes,
                 optimizer,
                 update_frequency,
                 save_dir):
        super(Worker, self).__init__()
        self.worker_index = worker_index
        self.name = f"Worker_{worker_index}"
        self.env = gym.make(gym_game_name)
        self.state_space = self.env.observation_space.shape[0]
        self.action_space = self.env.action_space.n
        self.max_episodes = max_episodes
        self.optimizer = optimizer
        self.update_frequency = update_frequency
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        self.global_network = global_network
        self.local_network = model.A3CNetwork(self.state_space, self.action_space)

    def run(self):
        history = History()
        update_counter = 0
        ep_reward = 0
        while Worker.global_episode < self.max_episodes:
            current_state = self.env.reset()
            history.clear()

            done = False
            while not done:
                action_log_prob, _ = self.local_network(
                    tf.convert_to_tensor(current_state[None, :], dtype=tf.float32)
                )
                action_prob = tf.nn.softmax(tf.squeeze(action_log_prob)).numpy()
                action = np.random.choice(self.action_space, p=action_prob)
                new_state, reward, done, _ = self.env.step(action)
                if done:
                    reward = -1
                ep_reward += reward

                history.append(current_state, action, reward)

                if update_counter == self.update_frequency or done:
                    with tf.GradientTape() as tape:
                        local_loss = self.local_network.get_loss(done, new_state, history)
                    local_gradients = tape.gradient(local_loss, self.local_network.trainable_variables)
                    self.optimizer.apply_gradients(
                        zip(local_gradients, self.global_network.trainable_variables)
                    )
                    self.local_network.set_weights(self.global_network.get_weights())

                    history.clear()
                    update_counter = 0

                update_counter += 1
                current_state = new_state

            if ep_reward > Worker.best_score:
                print(f"New best score of {ep_reward} achieved by Worker {self.name}!")
                with Worker.save_lock:
                    self.global_network.save_weights(
                        os.path.join(
                            self.save_dir,
                            'best_model.h5'
                        )
                    )
                    print(f"Saved best model at: {os.path.join(self.save_dir, 'best_model.h5')}")
                Worker.best_score = ep_reward
            ep_reward = 0
            Worker.global_episode += 1


class TestWorker(threading.Thread):
    def __init__(self,
                 global_network,
                 gym_game_name,
                 max_episodes,
                 test_dir,
                 render=True):
        super(TestWorker, self).__init__()
        self.global_network = global_network
        self.gym_game_name = gym_game_name
        self.env = gym.make(gym_game_name)
        self.state_space = self.env.observation_space.shape[0]
        self.action_space = self.env.action_space.n
        self.max_episodes = max_episodes
        self.test_dir = test_dir
        os.makedirs(test_dir, exist_ok=True)
        self.render = render

    def run(self):
        episode = 0
        best_reward = 0
        average_reward = 0
        while episode < self.max_episodes:
            current_state = self.env.reset()
            ep_reward = 0
            done = False
            while not done:
                self.env.render()
                action_log_prob, _ = self.global_network(
                    tf.convert_to_tensor(current_state[None, :], dtype=tf.float32)
                )
                action_prob = tf.nn.softmax(tf.squeeze(action_log_prob)).numpy()
                action = np.random.choice(self.action_space, p=action_prob)
                new_state, reward, done, _ = self.env.step(action)

                if done:
                    reward = -1
                ep_reward += reward

                current_state = new_state
            if ep_reward > best_reward:
                best_reward = ep_reward
            average_reward += ep_reward
            episode += 1
        average_reward /= self.max_episodes
        with open(self._get_filepath(), "w+") as fp:
            fp.write(f"Best Reward: {best_reward}\n")
            fp.write(f"Average Reward: {average_reward}\n")

