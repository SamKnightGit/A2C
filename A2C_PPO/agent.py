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
                 num_checkpoints,
                 norm_clip_value,
                 optimizer,
                 random_seed,
                 save_dir):
        self.network = network
        self.workers = []
        for worker_index in range(num_workers):
            worker = Worker(
                worker_index, 
                gym_game_name,
                timesteps_per_episode,
                network,
                timesteps_per_rollout,
                random_seed
            )
            self.workers.append(worker)
        self.timesteps_per_rollout = timesteps_per_rollout
        self.timesteps_per_episode = timesteps_per_episode
        self.num_episodes = num_episodes
        self.episodes_per_checkpoint = int(num_episodes / num_checkpoints)
        self.norm_clip_value = norm_clip_value
        self.optimizer = optimizer
        self.save_dir = save_dir

    def run(self):
        rollouts_per_episode = int(self.timesteps_per_episode / self.timesteps_per_rollout)

        for episode in range(self.num_episodes):
            current_checkpoint = int(episode / self.episodes_per_checkpoint)
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
                    if done:
                        print(f"Worker {worker.worker_index} finished. Resetting environment!")
                        worker.reset_env()
                    if not os.path.exists(os.path.join(self.save_dir, f"checkpoint_{current_checkpoint}.h5")):
                        self.network.save_weights(
                            os.path.join(self.save_dir, f"checkpoint_{current_checkpoint}.h5")
                        )

class Worker:
    def __init__(self,
                 worker_index,
                 gym_game_name,
                 timesteps_per_episode,
                 network,
                 timesteps_per_rollout,
                 random_seed=None):
        self.worker_index = worker_index
        self.name = f"Worker_{worker_index}"
        self.env = gym.make(gym_game_name)
        self.env._max_episode_steps = timesteps_per_episode
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
                action_prob, _ = self.network(
                    tf.convert_to_tensor(current_state[np.newaxis, :], dtype=tf.float32)
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
                fp.write("Best Reward:".ljust(20) + f"{best_reward}\n")
                fp.write("Average Reward:".ljust(20) + f"{average_reward}\n")

