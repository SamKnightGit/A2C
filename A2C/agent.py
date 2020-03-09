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
                 number_envs, 
                 gym_game_name,
                 timesteps_per_rollout, 
                 timesteps_per_episode, 
                 num_episodes,
                 num_checkpoints,
                 norm_clip_value,
                 discount_factor,
                 optimizer,
                 random_seed,
                 save_dir,
                 summary_writer):
        self.network = network
        self.timesteps_per_rollout = timesteps_per_rollout
        self.timesteps_per_episode = timesteps_per_episode
        self.env = ParallelEnvironment(gym_game_name, number_envs, network, timesteps_per_rollout, discount_factor, random_seed)
        self.num_episodes = num_episodes
        self.episodes_per_checkpoint = int(num_episodes / num_checkpoints)
        self.norm_clip_value = norm_clip_value
        self.optimizer = optimizer
        self.save_dir = save_dir
        self.summary_writer = summary_writer
        self.smoothed_reward = []

    def add_to_smoothed_reward(self, reward):
        if len(self.smoothed_reward) == 0:
            self.smoothed_reward.append(reward)
        else:
            self.smoothed_reward.append(0.01 * reward + 0.99 * self.smoothed_reward[-1])


    def run(self):
        rollouts_per_episode = int(self.timesteps_per_episode / self.timesteps_per_rollout)

        for episode in range(self.num_episodes):
            current_checkpoint = int(episode / self.episodes_per_checkpoint)
            for rollout in range(rollouts_per_episode):
                states, actions, rewards, dones, values = self.env.get_trajectories()
                print(f"Rewards: {rewards}")
                with tf.GradientTape() as tape:
                    loss = self.network.get_loss(states, rewards, dones, actions, values)
                print(f"Loss: {loss}")
                gradients = tape.gradient(loss, self.network.trainable_weights)
                print(f"Max reward at episode {episode}, rollout {rollout}: {max(rewards)}")

                
                if self.norm_clip_value:
                    gradients, _ = tf.clip_by_global_norm(gradients, self.norm_clip_value)
                self.optimizer.apply_gradients(
                    zip(gradients, self.network.trainable_weights)
                )
                # Average over rewards produced from trajectory.    
                self.add_to_smoothed_reward(tf.reduce_mean(rewards))
                    
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
        self.ep_reward = 0

    def reset_env(self):
        self.state = np.ravel(self.env.reset())
        self.ep_reward = 0

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
            self.ep_reward += reward
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


class ParallelEnvironment:
    def __init__(self, environment_name, number_environments, ac_network, rollout_length, gamma, random_seed=None):
        self.number_environments = number_environments
        self.envs = [gym.make(environment_name) for _ in range(number_environments)]
        self.ac_network = ac_network
        self.rollout_length = rollout_length
        self.gamma = gamma
        if random_seed:
            for env_index in range(len(self.envs)):
                self.envs[env_index].seed(random_seed + env_index)
        self.dones = [False for _ in range(number_environments)]
        
    def set_done(self, dones):
        done_indices = []
        for done_index in range(len(dones)):
            if dones[done_index]:
                done_indices.append(done_index)
        
        false_counter = 0
        for done_index in range(len(self.dones)):
            if self.dones[done_index] == False:
                if done_index in done_indices:
                    self.dones[done_index] = True
                    false_counter += 1

    def step(self, actions):
        actions = tf.squeeze(actions)
        print(actions)
        print(actions[0].numpy())
        steps = []
        for env_index in range(len(self.envs)):
            if self.dones[env_index] == True:
                continue
            steps.append(self.envs[env_index].step(actions[env_index].numpy()))
        states, rewards, dones, infos = zip(*steps)
        return np.stack(states), np.stack(rewards), np.stack(dones), infos
            

    def reset(self):
        self.dones = [False for _ in range(self.number_environments)]
        return np.stack([env.reset() for env in self.envs])

    def discount(self, rewards, dones, gamma):
        discounted_reward = []
        final_reward = 0
        for reward, done in zip(rewards[::-1], dones[::-1]):
            final_reward = reward + gamma * final_reward * (1.0 - done)
            discounted_reward.append(final_reward)
        return discounted_reward[::-1]

    def get_trajectories(self):
        states, actions, rewards, values, dones = [], [], [], [], []

        env_states = self.reset()
        for _ in range(self.rollout_length):
            trajectory_info = self.ac_network.act(env_states)
            env_actions = trajectory_info.action
            env_states, env_rewards, env_dones, _ = self.step(env_actions)

            states.append(env_states)
            actions.append(env_actions)
            rewards.append(env_rewards)
            values.append(trajectory_info.value)
            dones.append(env_dones)

            self.set_done(env_dones)
            if np.all(self.dones):
                break 
        

        states = np.array(states)
        actions = np.array(actions)
        rewards = np.array(rewards, dtype='float32')
        values = np.array(values, dtype='float32')
        dones = np.array(dones, dtype='bool')

        if self.gamma > 0:
            _, last_values = self.ac_network(env_states)
            for n, (past_rewards, past_dones, past_values) in enumerate(zip(rewards, dones, last_values)):
                past_rewards = past_rewards.tolist()
                past_dones = past_dones.tolist()

                if past_dones[-1] == 0:
                    past_rewards = self.discount(past_rewards + [past_values], past_dones+[0], self.gamma)[:-1]
                else:
                    past_rewards = self.discount(past_rewards, past_dones, self.gamma)
                rewards[n] = past_rewards 


        return states, actions, rewards, dones, values


