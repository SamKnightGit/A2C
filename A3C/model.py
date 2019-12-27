import tensorflow as tf
import numpy as np
from tensorflow.keras import layers
from collections import deque


class A3CNetwork(tf.keras.Model):
    def __init__(self, state_space, action_space, value_weight=0.5, entropy_coefficient=0.01):
        super(A3CNetwork, self).__init__()
        self.state_space = state_space
        self.action_space = action_space
        self.value_weight = value_weight
        self.entropy_coefficient = entropy_coefficient

        # self.dense_shared_1 = layers.Dense(100, activation='relu')
        self.dense_value_hidden = layers.Dense(100, activation='relu')
        self.dense_critic_hidden = layers.Dense(100, activation='relu')
        self.policy_log_odds = layers.Dense(self.action_space)
        self.value = layers.Dense(1)
        # Initialize network weights with random input
        self(tf.convert_to_tensor(np.random.random((1, self.state_space)), dtype=tf.float32))

    def call(self, inputs):
        policy_output = self.dense_critic_hidden(inputs)
        log_odds = self.policy_log_odds(policy_output)

        value_output = self.dense_value_hidden(inputs)
        value = self.value(value_output)

        return log_odds, value

    def get_loss(self, done, new_state, history, discount_factor=0.99):
        if done:
            estimated_reward = 0
        else:
            _, estimated_reward = self(tf.convert_to_tensor(new_state[None, :], dtype=tf.float32))

        all_discounted_rewards = deque()
        for reward in history.rewards[::-1]:
            discounted_reward = estimated_reward * discount_factor
            discounted_reward += reward
            all_discounted_rewards.appendleft(discounted_reward)

        action_log_probs, values = self(tf.convert_to_tensor(np.vstack(history.states), dtype=tf.float32))

        advantage = tf.convert_to_tensor(np.array(all_discounted_rewards)[:, None], dtype=tf.float32) - values

        value_loss = tf.square(advantage)

        policy = tf.nn.softmax(action_log_probs)
        entropy = tf.nn.softmax_cross_entropy_with_logits(policy, action_log_probs)
        policy_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(history.actions, action_log_probs)
        policy_loss *= tf.stop_gradient(advantage)
        policy_loss -= self.entropy_coefficient * entropy

        total_loss = tf.reduce_mean((policy_loss + self.value_weight * value_loss))
        return total_loss







