import os
import subprocess
os.chdir(os.path.dirname(os.path.dirname(__file__)))

if __name__ == "__main__":
    for num_workers in [1, 2, 4, 8, 16]:
        model_dir_path = f"./experiment/worker_test/workers_{num_workers}"
        os.makedirs(model_dir_path, exist_ok=True)
        subprocess.call([
            "python",
            "run.py",
            "--max_episodes=200",
            f"--num_workers={num_workers}",
            "--random_seed=222",
            "--num_checkpoints=10",
            "--test_episodes=100",
            f"--model_directory={model_dir_path}"
        ])
