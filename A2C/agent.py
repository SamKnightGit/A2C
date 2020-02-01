import threading
import tensorflow as tf
import model
import numpy as np
import os
import copy
import gym


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

class Coordinator:
    def __init__(self, 
                 network, 
                 num_workers, 
                 gym_game_name,
                 timesteps_per_rollout, 
                 timesteps_per_episode, 
                 num_episodes,
                 norm_clip_value,
                 optimizer,
                 random_seed):
        self.network = network
        self.workers = []
        for worker_index in range(num_workers):
            worker = Worker(
                worker_index, 
                gym_game_name,
                network,
                timesteps_per_rollout,
                random_seed
            )
            self.workers.append(worker)
        self.timesteps_per_rollout = timesteps_per_rollout
        self.timesteps_per_episode = timesteps_per_episode
        self.num_episodes = num_episodes
        self.norm_clip_value = norm_clip_value
        self.optimizer = optimizer

    def run(self):
        rollouts_per_episode = self.timesteps_per_episode / self.timesteps_per_rollout

        for episode in range(self.num_episodes):
            for rollout in range(rollouts_per_episode):
                for worker in self.workers:
                    print(f"Processing in worker {worker.worker_index}, rollout {rollout} of episode {episode}")
                    done, new_state, history = worker.work()
                    with tf.GradientTape() as tape:
                        loss = self.network.get_loss(done, new_state, history)
                    gradients = tape.gradient(loss, self.network.trainable_weights)
                    if self.norm_clip_value:
                        gradients, _ = tf.clip_by_global_norm(gradients, self.norm_clip_value)
                    self.optimizer.apply_gradients(
                        zip(gradients, self.network.trainable_weights)
                    )

class Worker:
    def __init__(self,
                 worker_index,
                 gym_game_name,
                 network,
                 timesteps_per_rollout,
                 random_seed=None):
        self.worker_index = worker_index
        self.name = f"Worker_{worker_index}"
        self.env = gym.make(gym_game_name)
        if random_seed is not None:
            self.env.seed(random_seed)
        self.state_space = self.env.observation_space.shape[0]
        self.action_space = self.env.action_space.n
        self.state = np.ravel(self.env.reset())
        self.network = network
        self.timesteps_per_rollout = timesteps_per_rollout

    def reset_env(self):
        self.state = np.ravel(self.env.reset())

    def work(self):
        history = History()
        timestep = 0
        current_state = self.state

        done = False
        while timestep < self.timesteps_per_rollout:
            action_prob, _ = self.network(
                tf.convert_to_tensor(current_state[np.newaxis, :], dtype=tf.float32)
            )
            action_prob = tf.squeeze(action_prob).numpy()

            action = np.random.choice(self.action_space, p=action_prob)
            new_state, reward, done, _ = self.env.step(action)

            new_state = np.ravel(new_state)
            if done:
                reward = -1

            history.append(current_state, action, reward)
            current_state = np.ravel(new_state)
            timestep += 1
        return done, current_state, history



class TestWorker:
    def __init__(self,
                 gym_game_name,
                 network,
                 max_episodes,
                 test_file_name,
                 render=True,
                 random_seed=None):
        super(TestWorker, self).__init__()
        self.network = network
        self.env = gym.make(gym_game_name)
        if random_seed is not None:
            self.env.seed(random_seed)
        self.state_space = self.env.observation_space_dim
        self.action_space = self.env.action_space_dim
        self.max_episodes = max_episodes
        self.test_file_name = test_file_name
        self.render = render

        self.reward_breakdown = []
        self.base_station_locations = []
        self.actions = []
        self.base_station_actions = []
        self.user_locations = []
        self.outage_fraction = []

    def _record_initial_info(self):
        self.base_station_locations.append(copy.deepcopy(self.env.bsLoc))
        self.user_locations.append(copy.deepcopy(self.env.ueLoc))

    def _record_info(self, info, real_action):
        self.reward_breakdown.append(info.r_dissect)
        self.base_station_locations.append(info.bs_loc)
        self.actions.append(real_action)
        self.base_station_actions.append(info.bs_actions)
        self.user_locations.append(info.ue_loc)
        self.outage_fraction.append(info.outage_fraction)

    def _save_info(self):
        test_dir = os.path.dirname(self.test_file_name)
        np.save(os.path.join(test_dir, "reward_breakdown"), self.reward_breakdown)
        np.save(os.path.join(test_dir, "base_station_locations"), self.base_station_locations)
        np.save(os.path.join(test_dir, "base_station_actions"), self.base_station_actions)
        np.save(os.path.join(test_dir, "instructed_actions"), self.actions)
        np.save(os.path.join(test_dir, "user_locations"), self.user_locations)
        np.save(os.path.join(test_dir, "outage_fraction"), self.outage_fraction)

    def run(self):
        episode = 0
        best_reward = 0
        average_reward = 0
        while episode < self.max_episodes:
            current_state = np.ravel(self.env.reset())
            self._record_initial_info()

            ep_reward = 0
            done = False
            while not done:
                if self.render:
                    self.env.render()
                action_prob, _ = self.network(
                    tf.convert_to_tensor(current_state[np.newaxis, :], dtype=tf.float32)
                )
                action_prob = tf.squeeze(action_prob).numpy()
                action = np.argmax(action_prob)

                new_state, reward, done, info = self.env.step_test(action)
                self._record_info(info, action)
                new_state = np.ravel(new_state)

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
                fp.write("Best Reward:".ljust(20) + f"{best_reward}\n")
                fp.write("Average Reward:".ljust(20) + f"{average_reward}\n")
        self._save_info()

