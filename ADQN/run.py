import gym
import agent
import model
import os
import tensorflow as tf
import numpy as np
import click
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from time import time, sleep
from tqdm import tqdm
from queue import Queue


@click.command()
@click.option('--env_name', type=str, default='CartPole-v1')
@click.option('--num_workers', type=int, default=16)
@click.option('--max_episodes', type=int, default=10000)
@click.option('--learning_rate', type=float, default=10e-4)
@click.option('--target_update_frequency', type=int, default=1)
@click.option('--network_update_frequency', type=int, default=20)
@click.option('--epsilon', type=float, default=0.10)
@click.option('--epsilon_annealing_strategy', type=str, default="linear")
@click.option('--annealing_episodes', type=int, default=2000)
@click.option('--discount_factor', type=float, default=0.99)
@click.option('--norm_clip_value', type=float, default=None)
@click.option('--num_checkpoints', type=int, default=10)
@click.option('--model_directory', type=click.Path(), default="")
@click.option('--test_model', type=bool, default=True)
@click.option('--test_episodes', type=int, default=50)
@click.option('--render_testing', type=bool, default=False)
@click.option('--random_seed', type=int, default=None)
@click.option('--logging_frequency', type=int, default=10)
@click.option('--save', type=bool, default=True)
def run_training(
        env_name,
        num_workers,
        max_episodes,
        learning_rate,
        target_update_frequency,
        network_update_frequency,
        epsilon,
        epsilon_annealing_strategy,
        annealing_episodes,
        discount_factor,
        norm_clip_value,
        num_checkpoints,
        model_directory,
        test_model,
        test_episodes,
        render_testing,
        random_seed,
        logging_frequency,
        save):
    env = gym.make(env_name)
    state_space = env.observation_space.shape[0]
    action_space = env.action_space.n

    if random_seed is not None:
        env.seed(random_seed)
        tf.random.set_seed(random_seed)
        np.random.seed(random_seed)

    main_network = model.ADQNetwork(
        state_space=state_space,
        action_space=action_space
    )

    target_network = model.ADQNetwork(
        state_space=state_space,
        action_space=action_space
    )
    target_network.set_weights(main_network.get_weights())

    if not model_directory:
        model_directory = os.path.join(
            "./experiment/",
            f"{env_name}_{datetime.now()}"
        )
    logging_directory = os.path.join(
        model_directory,
        "logs"
    )
    summary_writer = tf.summary.create_file_writer(logging_directory)

    if save:
        os.makedirs(model_directory, exist_ok=True)
        os.makedirs(logging_directory, exist_ok=True)
    epsilon_minimum_values = [0.1, 0.01, 0.5]
    reward_queue = Queue()
    optimizer = tf.optimizers.Adam(learning_rate=learning_rate)
    workers = [
        agent.Worker(
            worker_index,
            env_name,
            random_seed,
            max_episodes,
            optimizer,
            target_network,
            main_network,
            target_update_frequency,
            network_update_frequency,
            epsilon_minimum_values[worker_index % len(epsilon_minimum_values)],
            #epsilon,
            epsilon_annealing_strategy,
            annealing_episodes,
            discount_factor,
            norm_clip_value,
            num_checkpoints,
            reward_queue,
            logging_frequency,
            summary_writer,
            model_directory,
            save
        ) for worker_index in range(num_workers)
    ]
    start_time = time()
    for worker in workers:
        print(f"Starting Worker: {worker.name}")
        worker.start()

    moving_average_rewards = []
    while True:
        reward = reward_queue.get()
        if reward is not None:
            moving_average_rewards.append(reward)
        else:
            break

    for worker in workers:
        worker.join()

    end_time = time()
    time_taken = end_time - start_time

    if save:
        write_summary(model_directory,
                      num_workers,
                      max_episodes,
                      learning_rate,
                      target_update_frequency,
                      network_update_frequency,
                      epsilon,
                      epsilon_annealing_strategy,
                      annealing_episodes,
                      discount_factor,
                      norm_clip_value,
                      time_taken,
                      random_seed,
                      main_network,
                      filename="summary.txt")
        plt.plot(moving_average_rewards)
        plt.ylabel('Moving average reward')
        plt.xlabel('Episode')
        plt.savefig(os.path.join(model_directory, 'Moving_Average.png'))

        if test_model:
            test_dir = os.path.join(model_directory, "test")
            os.makedirs(test_dir)
            print("Running tests with checkpoint policies...")
            for checkpoint in tqdm(range(num_checkpoints + 1)):
                if checkpoint == num_checkpoints:
                    checkpoint = "best"

                model_file_path = os.path.join(
                    model_directory,
                    f"checkpoint_{checkpoint}.h5"
                )
                if not os.path.exists(model_file_path):
                    break

                test_file_path = os.path.join(
                    test_dir,
                    f"test_checkpoint_{checkpoint}.txt"
                )

                run_testing(
                    env_name,
                    test_episodes,
                    model_file_path,
                    test_file_path,
                    render_testing
                )


def run_testing(
        env_name,
        max_episodes,
        model_file,
        test_file_name,
        render):

    env = gym.make(env_name)
    state_space = env.observation_space.shape[0]
    action_space = env.action_space.n
    global_network = model.ADQNetwork(
        state_space=state_space,
        action_space=action_space
    )

    global_network.load_weights(
        model_file
    )

    worker = agent.TestWorker(
        global_network,
        env_name,
        max_episodes,
        test_file_name,
        render=render
    )
    worker.start()
    worker.join()


def write_summary(
        model_directory,
        num_workers,
        max_episodes,
        learning_rate,
        target_update_frequency,
        network_update_frequency,
        epsilon,
        epsilon_annealing_strategy,
        annealing_episodes,
        discount_factor,
        norm_clip_value,
        time_taken,
        random_seed,
        main_network,
        filename="summary.txt"):
    filepath = os.path.join(model_directory, filename)
    with open(filepath, "w+") as fp:
        fp.write("Number of Workers:".ljust(35) + f"{num_workers}\n")
        fp.write("Training Episodes:".ljust(35) + f"{max_episodes}\n")
        fp.write("Learning Rate:".ljust(35) + f"{learning_rate}\n")
        fp.write("Target Update Frequency (eps):".ljust(35) + f"{target_update_frequency}\n")
        fp.write("Network Update Frequency:".ljust(35) + f"{network_update_frequency}\n")
        fp.write("Minimum Epsilon Values:".ljust(35) + f"{epsilon_minimum_values}\n")
        fp.write("Epsilon Annealing Strategy:".ljust(35) + f"{epsilon_annealing_strategy}\n")
        fp.write("Epsilon Annealing Episodes:".ljust(35) + f"{annealing_episodes}\n")
        fp.write("Discount Factor:".ljust(35) + f"{discount_factor}\n")
        fp.write("Norm Clip Value:".ljust(35) + f"{norm_clip_value}\n")
        fp.write("Time Taken:".ljust(35) + f"{time_taken}\n")
        fp.write("Formatted Time:".ljust(35) + f"{timedelta(seconds=time_taken)}\n")
        fp.write("Random Seed:".ljust(35) + f"{random_seed}\n")
        fp.write("Network Architecture:\n")
        main_network.summary(print_fn=lambda summ: fp.write(summ + "\n"))


if __name__ == "__main__":
    run_training()


