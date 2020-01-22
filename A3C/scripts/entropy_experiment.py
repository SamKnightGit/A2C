import os
import subprocess
os.chdir(os.path.dirname(os.path.dirname(__file__)))

if __name__ == "__main__":
    for entropy in [0.05, 0.5, 1, 5, 10]:
        model_dir_path = f"./experiment/entropy_test/entropy_{entropy}"
        os.makedirs(model_dir_path, exist_ok=True)
        subprocess.call([
            "python",
            "run.py",
            "--env_name=Acrobot-v1",
            f"--entropy_coefficient={entropy}",
            "--max_episodes=2000",
            f"--num_workers=10",
            "--num_checkpoints=10",
            "--test_episodes=10",
            f"--model_directory={model_dir_path}"
        ])
