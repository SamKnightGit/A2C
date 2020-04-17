import os
os.chdir(os.path.dirname(os.path.dirname(__file__)))
import unittest
import gym
import numpy as np
from model import A2CNetwork
from agent import ParallelEnvironment


class TestParallelEnvironment(unittest.TestCase):
    def setUp(self):
        env = gym.make("CartPole-v0")
        state_space = env.observation_space.shape[0]
        action_space = env.action_space.n
        self.network = A2CNetwork(state_space, action_space)
        self.parallel_env = ParallelEnvironment("CartPole-v0", 4, self.network, 100, 0.99)
    
    def test_step(self):
        states = self.parallel_env.reset()
        assert len(states) == 4
        info = self.network.act(states)
        
        new_states = self.parallel_env.step(info.action)
        assert not np.array_equal(states, new_states)

    def test_dones(self):
        assert self.parallel_env.dones == [False, False, False, False]
        dones = [False, True, False, False]
        self.parallel_env.set_done(dones)
        assert self.parallel_env.dones == [False, True, False, False]
        dones = [True, False, True]
        self.parallel_env.set_done(dones)
        assert self.parallel_env.dones == [True, True, False, True]
    


if __name__ == "__main__":
    unittest.main()