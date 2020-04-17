#!/bin/bash

python run.py --env_name="CartPole-v0" --num_workers=8 --max_episodes=2000 --target_update_frequency=1 --annealing_episodes=200 --discount_factor=0.99 --learning_rate=1e-2 --norm_clip_value=1 --random_seed=10 --model_directory="/home/sam/Documents/Dissertation/gym/ADQN/experiment/random_seed_cartpolev0_lr01/10"

python run.py --env_name="CartPole-v0" --num_workers=8 --max_episodes=2000 --target_update_frequency=1 --annealing_episodes=200 --discount_factor=0.99 --learning_rate=1e-2 --norm_clip_value=1 --random_seed=20 --model_directory="/home/sam/Documents/Dissertation/gym/ADQN/experiment/random_seed_cartpolev0_lr01/20"

python run.py --env_name="CartPole-v0" --num_workers=8 --max_episodes=2000 --target_update_frequency=1 --annealing_episodes=200 --discount_factor=0.99 --learning_rate=1e-2 --norm_clip_value=1 --random_seed=30 --model_directory="/home/sam/Documents/Dissertation/gym/ADQN/experiment/random_seed_cartpolev0_lr01/30"

python run.py --env_name="CartPole-v0" --num_workers=8 --max_episodes=2000 --target_update_frequency=1 --annealing_episodes=200 --discount_factor=0.99 --learning_rate=1e-2 --norm_clip_value=1 --random_seed=40 --model_directory="/home/sam/Documents/Dissertation/gym/ADQN/experiment/random_seed_cartpolev0_lr01/40"
