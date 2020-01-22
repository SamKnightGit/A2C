import multiprocessing
import threading
import gym
import tensorflow as tf
import model
import numpy as np
import os
import math
from queue import Queue
from datetime import datetime
NUM_CORES = multiprocessing.cpu_count()

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


class Worker(multiprocessing.Process):
    global_episode = 0
    global_average_running_reward = 0
    best_score = 0
    best_checkpoint_score = 0
    save_lock = multiprocessing.Lock()
    checkpoint_lock = multiprocessing.Lock()
    update_lock = multiprocessing.Lock()

    def __init__(self,
                 worker_index,
                 global_network_weights,
                 global_network_weights_shape,
                 gym_game_name,
                 max_episodes,
                 optimizer,
                 update_frequency,
                 num_checkpoints,
                 reward_queue,
                 gradients_queue,
                 global_episode,
                 global_average_running_reward,
                 best_score,
                 best_checkpoint_score,
                 save_lock,
                 update_lock,
                 save_dir,
                 save=True):
        super(Worker, self).__init__()
        self.worker_index = worker_index
        self.name = f"Worker_{worker_index}"
        self.env = gym.make(gym_game_name)
        self.state_space = self.env.observation_space.shape[0]
        self.action_space = self.env.action_space.n
        self.max_episodes = max_episodes
        self.optimizer = optimizer
        self.update_frequency = update_frequency
        self.episodes_per_checkpoint = int(max_episodes / num_checkpoints)
        self.reward_queue = reward_queue
        self.gradients_queue = gradients_queue
        self.global_episode = global_episode
        self.global_average_running_reward = global_average_running_reward
        self.best_score = best_score
        self.best_checkpoint_score = best_checkpoint_score
        self.save_lock = save_lock
        self.update_lock = update_lock
        self.save_dir = save_dir
        self.save = save
        if save:
            os.makedirs(save_dir, exist_ok=True)
        self.global_network_weights = global_network_weights
        self.global_network_weights_shape = global_network_weights_shape
        self.local_network = model.A3CNetwork(self.state_space, self.action_space)

        self.gradients = []

    def _save_gradients(self, file_path):
        if self.save:
            np.save(file_path, self.gradients)
            self.gradients = []

    def _get_next_episode(self):
        with self.global_episode.get_lock():
            episode = self.global_episode.value
            self.global_episode.value += 1
        return episode

    def _update_weights(self):
        with self.global_network_weights.get_lock():
            weights = np.frombuffer(self.global_network_weights.array, dtype='double')
            shaped_weigts = []
            accumulated_index = 0
            for shape in self.global_network_weights_shape:
                
            self.global_network_weights_shape
        self.local_network.set_weights(weights)

    def run(self):
        history = History()
        update_counter = 0
        ep_reward = 0
        global_episode = self._get_next_episode()
        while global_episode < self.max_episodes:
            print(f"Starting global episode: {global_episode}")
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
                    local_gradients = tape.gradient(local_loss, self.local_network.trainable_weights)
                    local_gradients, _ = tf.clip_by_global_norm(local_gradients, 5.0)
                    if self.worker_index == 0:
                        self.gradients.append(local_gradients)
                    self.gradients_queue.put(local_gradients)
                    with self.update_lock:
                        self._update_weights()

                    history.clear()
                    update_counter = 0

                update_counter += 1
                current_state = new_state

            current_checkpoint = int(global_episode / self.episodes_per_checkpoint)
            checkpoint_path = os.path.join(self.save_dir, f"checkpoint_{current_checkpoint}.h5")
            gradient_path = os.path.join(self.save_dir, f"checkpoint_{current_checkpoint}_gradients")
            
            with self.save_lock:
                if ep_reward >= self.best_checkpoint_score.value:
                    if ep_reward >= self.best_score.value:
                        print(f"New global best score of {ep_reward} achieved by Worker {self.name}!")
                        self.best_score.value = ep_reward
                    print(f"New checkpoint best score of {ep_reward} achieved by Worker {self.name}!")
                    self.best_checkpoint_score.value = ep_reward

                if self.save and not os.path.exists(checkpoint_path):
                    self._save_global_weights(f"checkpoint_{current_checkpoint}.h5")
                    Worker.best_checkpoint_score = 0

                if self.worker_index == 0 and (not os.path.exists(gradient_path)):
                    self._save_gradients(gradient_path)

                if self.global_average_running_reward.value == 0:
                    self.global_average_running_reward.value = ep_reward
                else:
                    self.global_average_running_reward.value = self.global_average_running_reward.value * 0.99 + ep_reward * 0.01
                self.reward_queue.put(self.global_average_running_reward.value)
            ep_reward = 0
            global_episode = self._get_next_episode()

        self.reward_queue.put(None)




class TestWorker(threading.Thread):
    def __init__(self,
                 global_network,
                 gym_game_name,
                 max_episodes,
                 test_file_name,
                 render=True):
        super(TestWorker, self).__init__()
        self.global_network = global_network
        self.gym_game_name = gym_game_name
        self.env = gym.make(gym_game_name)
        self.state_space = self.env.observation_space.shape[0]
        self.action_space = self.env.action_space.n
        self.max_episodes = max_episodes
        self.test_file_name = test_file_name
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
                if self.render:
                    self.env.render()
                action_log_prob, _ = self.global_network(
                    tf.convert_to_tensor(current_state[None, :], dtype=tf.float32)
                )
                action_prob = tf.nn.softmax(tf.squeeze(action_log_prob)).numpy() + 1e-9
                action = np.argmax(action_prob)
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
        if self.test_file_name:
            with open(self.test_file_name, "w+") as fp:
                fp.write(f"Best Reward: {best_reward}\n")
                fp.write(f"Average Reward: {average_reward}\n")

