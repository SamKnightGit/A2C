import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
from tensorflow.keras import layers, initializers, losses
from collections import deque, namedtuple


class A2CNetwork(tf.keras.Model):
    def __init__(self, state_space, action_space, value_weight=0.5, entropy_coefficient=0.01):
        super(A2CNetwork, self).__init__()
        self.state_space = state_space
        self.action_space = action_space
        self.value_weight = value_weight
        self.entropy_coefficient = entropy_coefficient

        # self.dense_shared_1 = layers.Dense(100, activation='relu')
        self.dense_actor_hidden = layers.Dense(100, activation='relu', input_dim=self.state_space,
                                               kernel_initializer=initializers.glorot_uniform)
        self.dense_critic_hidden = layers.Dense(100, activation='relu', input_dim=self.state_space,
                                                kernel_initializer=initializers.glorot_uniform)
        self.policy = layers.Dense(self.action_space, activation=tf.nn.softmax,
                                   kernel_initializer=initializers.glorot_uniform)
        self.value = layers.Dense(1, kernel_initializer=initializers.glorot_uniform)
        # Initialize network weights with random input
        self(tf.convert_to_tensor(np.random.random((1, self.state_space)), dtype=tf.float32))

    def call(self, inputs):
        policy_output = self.dense_actor_hidden(inputs)
        policy = self.policy(policy_output)

        value_output = self.dense_critic_hidden(inputs)
        value = self.value(value_output)

        return policy, value

    def act(self, state, action=None):
        policy, value = self(tf.convert_to_tensor(state[None, :], dtype=tf.float32))
        policy_distribution = tfp.distributions.Categorical(probs=policy)
        if action is None:
            action = policy_distribution.sample()

        neg_log_probability = -policy_distribution.log_prob(action)
        entropy = policy_distribution.entropy()
        ActionInfo = namedtuple('ActionInfo', 'action neg_log_prob entropy value')
        info = ActionInfo(action, neg_log_probability, entropy, tf.squeeze(value))
        return info

    def get_loss(self, states, rewards, dones, actions, values):
        advantages = rewards - values
        action_info = self.act(states, actions)

        actor_loss = tf.reduce_mean(action_info.neg_log_prob * advantages)
        entropy = tf.reduce_mean(action_info.entropy)
        critic_loss = tf.reduce_mean(tf.square(action_info.value, rewards))

        actor_critic_loss = actor_loss + self.value_weight * critic_loss 
        total_loss = tf.subtract(actor_critic_loss, self.entropy_coefficient * entropy)
        return total_loss

    # def train(self, rewards, actions, logprobs, values, entropy):
    #     # Advantage
    #     advantages = rewards - values

    #     action_indices = np.zeros((actions.shape[0], 2))
    #     action_indices[:,0] = np.arange(actions.shape[0])
    #     action_indices[:,1] = actions
    #     action_indices = tf.convert_to_tensor(action_indices, dtype=tf.int32)
    #     neglogprob_given_actions = tf.gather_nd(-logprobs, action_indices)
        
    #     loss = self.get_loss(neglogprob_given_actions, advantages, entropy)


