import tensorflow as tf
import numpy as np
from tensorflow.keras import layers, initializers
from collections import deque


class ADQNetwork(tf.keras.Model):
    def __init__(self, state_space, action_space):
        super(ADQNetwork, self).__init__()
        self.state_space = state_space
        self.action_space = action_space

        self.dense_1 = layers.Dense(400, activation='relu', input_dim=self.state_space,
                                           kernel_initializer=initializers.glorot_uniform)
        self.dense_2 = layers.Dense(400, activation='relu', kernel_initializer=initializers.glorot_uniform)
        # self.dense_value_hidden = layers.Dense(100, activation='relu', input_dim=self.state_space,
        #                                        kernel_initializer=initializers.glorot_uniform)
        # self.dense_critic_hidden = layers.Dense(100, activation='relu', input_dim=self.state_space,
        #                                         kernel_initializer=initializers.glorot_uniform)
        self.q_value = layers.Dense(self.action_space, activation=tf.nn.softmax,
                                    kernel_initializer=initializers.glorot_uniform)
        # Initialize network weights with random input
        self(tf.convert_to_tensor(np.random.random((1, self.state_space)), dtype=tf.float32))

    def call(self, inputs):
        output = self.dense_1(inputs)
        output = self.dense_2(output)

        q_value = self.q_value(output)

        return q_value

    def get_loss(self, history):
        targets = np.array(history.targets)
        states = np.array(history.states)
        states = np.expand_dims(states, axis=0)
        actions = np.array(history.actions)
        action_indices = np.zeros((actions.shape[0], 2))
        action_indices[:,0] = np.arange(actions.shape[0])
        action_indices[:,1] = actions
        action_indices = tf.convert_to_tensor(action_indices, dtype=tf.int32)
        
        action_prob = self(tf.convert_to_tensor(states[None, :], dtype=tf.float32))
        action_prob = tf.squeeze(action_prob)
        number_experiences = targets.shape[0] if len(targets.shape) > 1 else 1
        if len(action_prob.shape) == 1: # Only one action taken
            action_prob = tf.expand_dims(action_prob, 0)
        total_loss = tf.reduce_sum(
            tf.square(targets - tf.gather_nd(action_prob, action_indices))
        )
        total_loss = tf.divide(total_loss, number_experiences)
        return total_loss








