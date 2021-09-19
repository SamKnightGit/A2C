## Reinforcement Learning Algorithms in Tensorflow 2
This repo contains implementations for a number of popular deep reinforcement algorithms, specifically, actor-critic and asynchronous deep Q learning algorithms. Algorithms are implemented in Tensorflow 2 and tested against [OpenAI's gym environment](https://gym.openai.com/). 

The aim of these implementations is to provide clear and simple references for their respective algorithms. I aimed to optimize the implementation for readability/comprehension and I would strongly recommend visiting [OpenAI's baseline repo](https://github.com/openai/baselines) for a broader range of reference implementations. 

## Project structure:
All algorithms are housed in their own directory with the following structure:
- model.py: Contains the neural network model used during training and evaluation
- agent.py: Contains the logic for each agent (worker) that is interacting with the environment. When necessary, provides classes for handling coordination between workers.
- run.py: Contains entry-point for training and testing the models.

Please find the sources for the implemented algorithms listed below:
- A3C & ADQN -- https://arxiv.org/abs/1602.01783
- A2C -- https://openai.com/blog/baselines-acktr-a2c/
- PPO -- https://arxiv.org/abs/1707.06347

## Pre-requisites
The following instructions have been tested with Python3.6 on Ubuntu 18.04

There are a number of ways to install Python3.6, but please note that certain dependencies may be missing when building from source. The following tutorials may be helpful for installing Python:
- [deadsnakes PPA](https://tooling.bennuttall.com/deadsnakes/) on Ubuntu
- [homebrew](https://docs.python-guide.org/starting/install3/osx/) on Mac OS X 

## Installation
Clone the repository and navigate into it

`cd gym`

`git clone https://github.com/SamKnightGit/AC.git`

Set up a virtual environment (or activate an existing virtual environment)

`pip install virtualenv`

`virtualenv /venv --python=python3`

`source /venv/bin/activate`

Install dependencies from the requirements file

`pip install -r requirements.txt`

## Running Experiments

The various models can be trained from the command line. Navigate into a specific algorithm's directory and explore the options for training / testing:

`python3 run.py --help`

The **env_name** is used to determine which of the gym environments the agents should interact with ([full list of environments](https://gym.openai.com/envs/#classic_control)).

When running an experiment with the **save** flag enabled, checkpoints of the model are stored in the path specified by **model_directory**. If no path is specified the following default path is used: 

`{current directory}/experiments/{gym environment}_{timestamp}`

Test checkpoints containing the average and best reward for each of the trained model checkpoints are evaluated for **test_episodes** on the environment. The results are saved in corresponding experiment directory if the **test_model** flag is enabled. These test runs can be visualized by enabling the **render_testing** flag.

The rest of these options are hyperparameters such as the **learning_rate** or number of workers to run concurrently -- **num_workers**. I encourage you to experiment with adjusting these options to see what affect they have on the performance of the trained model.



