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
    """
    Main function to set up the environment and train the RL model.
    """
    env = setup_mario_environment()

    # Define directories for model checkpoints and logs
    CHECKPOINT_DIR = '../AleProjet/checkpoints'
    LOG_DIR = '../AleProjet/logs'

    # Instantiate the saving model callback
    # Reduced check_freq to save more often relative to the new total_timesteps
    callback = TrainAndLoggingCallback(check_freq=1000, save_path=CHECKPOINT_DIR)

    # Instantiate the PPO model
    # 'CnnPolicy' is suitable for image-based observations
    model = PPO(
        'CnnPolicy',
        env,
        verbose=1, # Set to 1 to see training progress in the console
        tensorboard_log=LOG_DIR,
        learning_rate=0.000001,
        n_steps=512 # Number of steps to run for each environment per update
    )

    # Train the RL model
    print("Starting model training...")
    # total_timesteps defines how many frames the agent will experience during training
    # Reduced total_timesteps significantly for faster execution
    model.learn(total_timesteps=10000, callback=callback)
    print("Model training finished.")

    model.save('thisisatestmodel')
    # Load model
    model = PPO.load('./checkpoints/best_model_19200')

    state = env.reset()

    # Start the game
    state = env.reset()
    # Loop through the game
    while True:
        action, _ = model.predict(state)
        state, reward, done, info = env.step(action)
        env.render()

    # Close the environment ONLY after training is complete
    env.close()


if __name__ == "__main__":
    main()
