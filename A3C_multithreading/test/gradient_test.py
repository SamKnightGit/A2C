import os
import numpy as np

os.chdir("/home/sam/Documents/Dissertation/gym/A3C")

gradients = np.load("./experiment/CartPole-v0_2020-01-03 13:44:35.507059/checkpoint_8_gradients.npy", allow_pickle=True)

print(gradients)