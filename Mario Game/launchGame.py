import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
import numpy as np
# Import frame stacker wrapper and gray-scale wrapper
from gym.wrappers import FrameStack, GrayScaleObservation

# Import Vectorization Wrapper
from stable_baselines3.common.vec_env import VecFrameStack, DummyVecEnv

# Import matplotlib (not used in the main training flow, but kept if user wants to re-add visualization)
import matplotlib.pyplot as plt

# NECESSARY LIBRARIES TO TRAIN THE RL MODEL
import os
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback


class TrainAndLoggingCallback(BaseCallback):
    def __init__(self, check_freq, save_path, verbose=1):
        super(TrainAndLoggingCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.save_path = save_path

    def _init_callback(self):
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self):
        if self.n_calls % self.check_freq == 0:
            model_path = os.path.join(self.save_path, 'best_model_{}'.format(self.n_calls))
            self.model.save(model_path)
        return True


def setup_mario_environment():
    """
    Sets up the Super Mario Bros environment with necessary wrappers.
    """
    env = gym_super_mario_bros.make('SuperMarioBros2-v1')
    env = JoypadSpace(env, SIMPLE_MOVEMENT)
    env = GrayScaleObservation(env, keep_dim=True)
    env = DummyVecEnv([lambda: env])
    env = VecFrameStack(env, 4, channels_order='last')
    return env


def main():
    env = setup_mario_environment()

    # Define directories for model checkpoints and logs (still needed if loading from these paths)
    CHECKPOINT_DIR = '../AleProjet/checkpoints'
    LOG_DIR = '../AleProjet/logs'



    loaded_model = PPO.load('thisisatestmodel')

    print("Starting game with loaded model...")
    # Reset the environment to start a new episode for evaluation
    state = env.reset()
    done = False
    total_reward = 0

    # Loop through the game until the episode is done
    while not done:
        # Predict action from the loaded model
        action, _states = loaded_model.predict(state)

        # Take a step in the environment
        state, reward, done, info = env.step(action)
        total_reward += reward

        # Render the environment to visualize the game
        env.render()

    print(f"Episode finished. Total Reward: {total_reward}")

    # Close the environment ONLY after evaluation is complete
    env.close()


if __name__ == "__main__":
    main()
