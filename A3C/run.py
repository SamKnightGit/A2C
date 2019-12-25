import gym
import agent
import model
import os
import tensorflow as tf
import click
from datetime import datetime


@click.command()
@click.option('--env_name', type=str, default='CartPole-v0')
@click.option('--num_workers', type=int, default=1)
@click.option('--max_episodes', type=int, default=100)
@click.option('--learning_rate', type=float, default=1e-3)
@click.option('--network_update_frequency', type=int, default=50)
@click.option('--training', type=bool, default=True)
@click.option('--model_directory', type=click.Path(), default="./train/")
@click.option('--test_directory', type=click.Path(), default="./test/")
@click.option('--test_file_name', type=str, default="")
def run_experiment(
        env_name,
        num_workers,
        max_episodes,
        learning_rate,
        network_update_frequency,
        training,
        model_directory,
        test_directory,
        test_file_name):

    env = gym.make(env_name)
    state_space = env.observation_space.shape[0]
    action_space = env.action_space.n
    global_network = model.A3CNetwork(
        state_space=state_space,
        action_space=action_space
    )

    if training:
        optimizer = tf.optimizers.Adam(learning_rate)
        workers = [
            agent.Worker(
                worker_index,
                global_network,
                env_name,
                max_episodes,
                optimizer,
                network_update_frequency,
                model_directory
            ) for worker_index in range(num_workers)
        ]
        for worker in workers:
            print(f"Starting Worker: {worker.name}")
            worker.start()

        for worker in workers:
            worker.join()

    else:  # testing
        global_network.load_weights(
            os.path.join(
                model_directory,
                'best_model.h5'
            )
        )

        test_file_name = _get_filepath(test_file_name, test_directory, env_name)

        worker = agent.TestWorker(
            global_network,
            env_name,
            max_episodes,
            test_file_name
        )
        worker.start()
        worker.join()


def _get_filepath(test_filename, test_directory, gym_game_name):
    current_datetime = str(datetime.now())
    if not test_filename:
        test_filename = gym_game_name + "_" + current_datetime
    return os.path.join(test_directory, test_filename)


if __name__ == "__main__":
    run_experiment()



