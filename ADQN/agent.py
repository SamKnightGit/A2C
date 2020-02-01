import threading
import gym
import tensorflow as tf
import model
import numpy as np
import os
import math
from queue import Queue
from datetime import datetime

class History:
    def __init__(self):
        self.states = []
        self.actions = []
        self.targets = []

    def append(self, state, action, target):
        self.states.append(state)
        self.actions.append(action)
        self.targets.append(target)

    def clear(self):
        self.states = []
        self.actions = []
        self.targets = []


class Worker(threading.Thread):
    global_episode = 0
    global_average_running_reward = 0
    best_score = 0
    best_checkpoint_score = 0
    save_lock = threading.Lock()
    checkpoint_lock = threading.Lock()
    update_lock = threading.Lock()

    def __init__(self,
                 worker_index,
                 gym_game_name,
                 random_seed,
                 max_episodes,
                 optimizer,
                 target_network,
                 main_network,
                 target_update_frequency,
                 update_frequency,
                 epsilon,
                 discount_factor,
                 norm_clip_value,
                 num_checkpoints,
                 reward_queue,
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
        self.target_network = target_network
        self.main_network = main_network
        self.target_update_frequency = target_update_frequency
        self.update_frequency = update_frequency
        self.epsilon = epsilon
        self.discount_factor = discount_factor
        self.norm_clip_value = norm_clip_value
        self.episodes_per_checkpoint = int(max_episodes / num_checkpoints)
        self.reward_queue = reward_queue
        self.save_dir = save_dir
        self.save = save
        if save:
            os.makedirs(save_dir, exist_ok=True)

        self.gradients = []

    def _save_gradients(self, file_path):
        if self.save:
            np.save(file_path, self.gradients)
            self.gradients = []

    def _save_global_weights(self, filename):
        if self.save:
            self.main_network.save_weights(
                os.path.join(
                    self.save_dir,
                    filename
                )
            )

    def _calculate_target(self, new_state, reward, done):
        if done:
            target_output = reward
        else:
            target_action_prob = self.target_network(
                tf.convert_to_tensor(new_state[None, :], dtype=tf.float32)
            )
            greedy_action_value = np.max(tf.squeeze(target_action_prob).numpy())
            target_output = reward + self.discount_factor * greedy_action_value
        return target_output

    def _get_next_episode(self):
        with Worker.checkpoint_lock:
            episode = Worker.global_episode
            Worker.global_episode += 1
        return episode

    def run(self):
        history = History()
        update_counter = 0
        ep_reward = 0
        global_episode = self._get_next_episode()
        while global_episode < self.max_episodes:
            print(f"Starting global episode: {global_episode}")
            if global_episode != 0 and global_episode % self.target_update_frequency == 0:
                print("Updating target network!")
                self.target_network.set_weights(self.main_network.get_weights())
            current_state = self.env.reset()
            self.gradients = []

            done = False
            while not done:
                if self.epsilon > np.random.random():
                    action = np.random.choice(self.action_space)
                else:
                    action_prob = self.main_network(
                        tf.convert_to_tensor(current_state[None, :], dtype=tf.float32)
                    )
                    action_prob = tf.squeeze(action_prob).numpy()
                    action = np.random.choice(self.action_space, p=action_prob)

                new_state, reward, done, _ = self.env.step(action)
                if done:
                    reward = -1
                ep_reward += reward
                history.append(current_state, action, self._calculate_target(new_state, reward, done))

                if update_counter == self.update_frequency or done:
                    with Worker.update_lock:
                        with tf.GradientTape() as tape:
                            local_loss = self.main_network.get_loss(history)
                        local_gradients = tape.gradient(local_loss, self.main_network.trainable_weights)
                        if self.norm_clip_value:
                            local_gradients, _ = tf.clip_by_global_norm(local_gradients, self.norm_clip_value)
                        if self.worker_index == 0:
                            self.gradients.append(local_gradients)
                        self.optimizer.apply_gradients(
                            zip(local_gradients, self.main_network.trainable_weights)
                        )

                    update_counter = 0

                update_counter += 1
                current_state = new_state

            current_checkpoint = int(global_episode / self.episodes_per_checkpoint)
            checkpoint_path = os.path.join(self.save_dir, f"checkpoint_{current_checkpoint}.h5")
            gradient_path = os.path.join(self.save_dir, f"checkpoint_{current_checkpoint}_gradients")

            with Worker.save_lock:
                if ep_reward >= Worker.best_checkpoint_score:
                    if ep_reward >= Worker.best_score:
                        print(f"New global best score of {ep_reward} achieved by Worker {self.name}!")
                        self._save_global_weights('checkpoint_best.h5')
                        print(f"Saved global best model at: {os.path.join(self.save_dir, 'checkpoint_best.h5')}")
                        print(f"Model Loss: {local_loss}")
                        Worker.best_score = ep_reward
                    print(f"New checkpoint best score of {ep_reward} achieved by Worker {self.name}!")
                    self._save_global_weights(f"checkpoint_{current_checkpoint}.h5")
                    print(f"Saved checkpoint best model at: {checkpoint_path}")
                    Worker.best_checkpoint_score = ep_reward

                if self.save and not os.path.exists(checkpoint_path):
                    self._save_global_weights(f"checkpoint_{current_checkpoint}.h5")
                    Worker.best_checkpoint_score = 0

                if self.worker_index == 0 and (not os.path.exists(gradient_path)):
                    self._save_gradients(gradient_path)

                if Worker.global_average_running_reward == 0:
                    Worker.global_average_running_reward = ep_reward
                else:
                    Worker.global_average_running_reward = Worker.global_average_running_reward * 0.99 + ep_reward * 0.01
            self.reward_queue.put(Worker.global_average_running_reward)
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
                action_prob = self.global_network(
                    tf.convert_to_tensor(current_state[None, :], dtype=tf.float32)
                )
                action_prob = tf.squeeze(action_prob).numpy()
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

