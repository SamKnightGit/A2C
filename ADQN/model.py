import tensorflow as tf
import numpy as np
from tensorflow.keras import layers, initializers
from collections import deque


class ADQNetwork(tf.keras.Model):
    def __init__(self, state_space, action_space):
        super(ADQNetwork, self).__init__()
        self.state_space = state_space
        self.action_space = action_space

        self.dense_shared_1 = layers.Dense(400, activation='relu', input_dim=self.state_space,
                                           kernel_initializer=initializers.glorot_uniform)
        self.dense_shared_2 = layers.Dense(400, activation='relu', input_dim=self.state_space,
                                           kernel_initializer=initializers.glorot_uniform)
        # self.dense_value_hidden = layers.Dense(100, activation='relu', input_dim=self.state_space,
        #                                        kernel_initializer=initializers.glorot_uniform)
        # self.dense_critic_hidden = layers.Dense(100, activation='relu', input_dim=self.state_space,
        #                                         kernel_initializer=initializers.glorot_uniform)
        self.q_value = layers.Dense(self.action_space, activation=tf.nn.softmax,
                                    kernel_initializer=initializers.glorot_uniform)
        # Initialize network weights with random input
        self(tf.convert_to_tensor(np.random.random((1, self.state_space)), dtype=tf.float32))

    def call(self, inputs):
        shared_output = self.dense_shared_1(inputs)
        shared_output = self.dense_shared_2(shared_output)

        q_value = self.q_value(shared_output)

        return q_value

    def get_loss(self, history):
        total_loss = 0
        for history_index in range(len(history.targets)):
            target = history.targets[history_index]
            state = history.states[history_index]
            action = history.actions[history_index]
            state = np.expand_dims(state, axis=0)
            print(state.shape)
            total_loss += target - self(tf.convert_to_tensor(np.expand_dims(state, axis=0), dtype=tf.float32))[action]
        return total_loss








