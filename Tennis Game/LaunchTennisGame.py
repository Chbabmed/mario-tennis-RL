import os

import gymnasium as gym
import ale_py
import numpy as np
import matplotlib.pyplot as plt

# Import stable-baselines3 components
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from gymnasium.wrappers import GrayscaleObservation
from stable_baselines3.common.vec_env import VecFrameStack, DummyVecEnv
from stable_baselines3.common.monitor import Monitor  # For logging episode rewards
from gymnasium import Wrapper  # Import Wrapper base class for custom wrappers
from stable_baselines3.common.preprocessing import get_obs_shape  # For observation normalization


# --- 1. Custom Reward Wrapper Definition ---
class CustomRewardWrapper(Wrapper):
    """
    A custom wrapper to modify the reward function
    """

    def __init__(self, env):
        super().__init__(env)
        print("CustomRewardWrapper initialized.")
        self.prev_observation = None
        self.prev_ball_pos = None
        self.prev_player_paddle_pos = None

    def _get_object_position(self, observation, color_threshold_low, color_threshold_high, y_range=None):
        """
        Helper function to find the approximate center position of an object
        """
        if observation.ndim == 3 and observation.shape[-1] == 1:
            img = observation[:, :, 0]  # 7ayad the last dim bach ykon (H,W)
        else:
            img = observation  # Assume it's already (H, W)

        mask = (img >= color_threshold_low) & (img <= color_threshold_high)

        # Apply Y-range filter if provided
        if y_range:
            y_min, y_max = y_range
            # Ensure y_min and y_max are within image bounds
            y_min = max(0, y_min)
            y_max = min(img.shape[0], y_max)
            # Create a sub-mask for the Y-range
            y_mask = np.zeros_like(mask, dtype=bool)
            y_mask[y_min:y_max, :] = True
            mask = mask & y_mask  # Combine with the color mask

        coords = np.argwhere(mask)

        if coords.shape[0] > 0:
            # Return the center of the detected pixels
            center_y, center_x = np.mean(coords, axis=0)
            return (center_x, center_y)  # Return (x, y) for consistency
        return None

    def detect_ball_hit(self, current_observation, info):
        """
        Conceptual function to detect if the agent's paddle has hit the ball.
        This is a heuristic based on pixel analysis and requires careful tuning.
        It's challenging to make robust without direct game state (RAM).
        """
        if self.prev_observation is None:
            self.prev_observation = current_observation
            return False

        # --- Refined Thresholds based on image_2578e5.png ---
        # Ball appears pure white
        BALL_COLOR_LOW = 240
        BALL_COLOR_HIGH = 255

        # Player paddle (bottom) and opponent paddle (top) are light gray
        PLAYER_PADDLE_COLOR_LOW = 170
        PLAYER_PADDLE_COLOR_HIGH = 200

        # Define Y-ranges to help distinguish objects
        # Assuming image height is around 210 pixels (from your plot)
        # Player paddle is in the bottom half
        PLAYER_PADDLE_Y_RANGE = (120, 180)  # Approximate Y-range for player's paddle
        # Ball can be anywhere in the court area
        BALL_Y_RANGE = (50, 180)  # Approximate Y-range for the ball in play

        # Get current positions with Y-range filtering
        current_ball_pos = self._get_object_position(current_observation, BALL_COLOR_LOW, BALL_COLOR_HIGH,
                                                     y_range=BALL_Y_RANGE)
        current_player_paddle_pos = self._get_object_position(current_observation, PLAYER_PADDLE_COLOR_LOW,
                                                              PLAYER_PADDLE_COLOR_HIGH, y_range=PLAYER_PADDLE_Y_RANGE)

        hit_detected = False

        if current_ball_pos and current_player_paddle_pos:
            # Calculate distance between ball and player's paddle
            distance = np.linalg.norm(np.array(current_ball_pos) - np.array(current_player_paddle_pos))

            # You'll need to determine a reasonable 'HIT_DISTANCE_THRESHOLD' by observation.
            # This is the maximum pixel distance at which we consider a "hit" to occur.
            HIT_DISTANCE_THRESHOLD = 8  # Example: pixels close

            # A hit is detected if the ball and paddle are very close.
            # This is a very rough heuristic. A more robust detection would also consider
            # the ball's velocity change or its movement direction relative to the paddle.
            if distance < HIT_DISTANCE_THRESHOLD:
                hit_detected = True

        # Update previous observations for next step
        self.prev_observation = current_observation
        self.prev_ball_pos = current_ball_pos
        self.prev_player_paddle_pos = current_player_paddle_pos

        return hit_detected

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)

        original_reward = reward
        modified_reward = original_reward

        # Amplify existing rewards (make wins/losses more impactful to learn more faster)
        if original_reward > 0:
            modified_reward += 0.1
        elif original_reward < 0:
            modified_reward -= 0.1

        # Penalize inactivity or specific undesirable actions
        if action == 0:  # Corrected: 'action' is a scalar here, no [0] needed
            modified_reward -= 0.0001  # penalty for not moving/acting

        # Reward for the ball hitting
        if self.detect_ball_hit(observation, info):
            modified_reward += 0.005  # Small reward for hitting the ball

        # Penalize player for going out of horizontal field ---
        FIELD_X_MIN = 10
        FIELD_X_MAX = 150
        OUT_OF_BOUNDS_PENALTY = 0.01

        PLAYER_PADDLE_COLOR_LOW = 170
        PLAYER_PADDLE_COLOR_HIGH = 200
        PLAYER_PADDLE_Y_RANGE = (120, 180)

        current_player_paddle_pos = self._get_object_position(
            observation, PLAYER_PADDLE_COLOR_LOW, PLAYER_PADDLE_COLOR_HIGH, y_range=PLAYER_PADDLE_Y_RANGE
        )

        if current_player_paddle_pos:
            player_x = current_player_paddle_pos[0]
            if player_x < FIELD_X_MIN or player_x > FIELD_X_MAX:
                modified_reward -= OUT_OF_BOUNDS_PENALTY
                print(f"Penalty: Player out of horizontal bounds at x={player_x}")

        modified_reward = float(modified_reward)

        return observation, modified_reward, terminated, truncated, info


# --- 2. Custom Callback for Saving Models and Logging ---
class TrainAndLoggingCallback(BaseCallback):
    """
    Custom callback for saving models during training and logging.
    """

    def __init__(self, check_freq: int, save_path: str, verbose: int = 1):
        super(TrainAndLoggingCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.save_path = save_path

    def _init_callback(self) -> None:
        """
        This method is called once when the callback is created.
        It ensures the save directory exists.
        """
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        """
        This method is called by the model after each call to env.step().
        It saves the model periodically.
        """
        if self.n_calls % self.check_freq == 0:
            model_path = os.path.join(self.save_path, f'best_model_{self.n_calls}')
            self.model.save(model_path)
            if self.verbose > 0:
                print(f"Saving model checkpoint at {model_path} (timesteps: {self.n_calls})")
        return True


# --- 3. Environment Setup Function ---
def setup_tennis_environment(render_mode=None, log_dir=None):
    """
    Sets up the Tennis environment with necessary wrappers for Stable Baselines3.
    """
    gym.register_envs(ale_py)

    env = gym.make('ALE/Tennis-v5', render_mode=render_mode)

    env = GrayscaleObservation(env, keep_dim=True)  # Capital 'S' here

    # Apply the CustomRewardWrapper AFTER grayscale conversion
    env = CustomRewardWrapper(env)

    if log_dir:
        env = Monitor(env, filename=os.path.join(log_dir, "monitor.csv"))

    env = DummyVecEnv([lambda: env])
    env = VecFrameStack(env, 4, channels_order='last')
    return env


# --- 4. Plotting Function for Training Results ---
def plot_results(log_dir: str):
    """
    Plots the episode rewards from the Monitor log file.
    """
    monitor_path = os.path.join(log_dir, "monitor.csv")
    if not os.path.exists(monitor_path):
        print(f"Monitor log file not found")
        return

    rewards = []
    try:
        with open(monitor_path, 'r') as f:
            lines = f.readlines()
        for line in lines[1:]:  # Skip header line
            parts = line.split(',')
            if len(parts) >= 1:
                try:
                    reward = float(parts[0])
                    rewards.append(reward)
                except ValueError:
                    continue  # Skip malformed lines
    except Exception as e:
        print(f"Error reading monitor.csv: {e}")
        return

    if not rewards:
        print("No rewards data found in monitor.csv to plot.")
        return

    # Plotting
    plt.figure(figsize=(12, 7))
    plt.plot(rewards)
    plt.title('PPO Training Scores for ALE/Tennis-v5')
    plt.xlabel('Episode Number')
    plt.ylabel('Total Reward')
    plt.grid(True)
    plt.show()


# --- 5. Main Function for Training and Evaluation ---
def main():
    # --- Define Directories for Checkpoints and Logs ---
    CHECKPOINT_DIR = './checkpoints'
    LOG_DIR = './logs'
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)  # Ensure directories exist
    os.makedirs(LOG_DIR, exist_ok=True)

    # --- Setup Environment for Training (COMMENTED OUT) ---
    # env = setup_tennis_environment(render_mode=None, log_dir=LOG_DIR)
    # print("Training environment setup complete.")

    # callback = TrainAndLoggingCallback(check_freq=10000, save_path=CHECKPOINT_DIR)
    # model = PPO(
    #     'CnnPolicy',
    #     env,
    #     verbose=1,
    #     tensorboard_log=LOG_DIR,
    #     learning_rate=0.000001,
    #     n_steps=512,
    #     device='cuda'
    # )
    # print("PPO model instantiated.")

    # # Train the RL model
    # TRAINING_TIMESTEPS = 500000
    # print(f"Starting model training for {TRAINING_TIMESTEPS} timesteps...")
    # model.learn(total_timesteps=TRAINING_TIMESTEPS, callback=callback)
    # print("Model training finished.")

    # # Save the final trained model
    # model.save('tennis_ppo_trained_model')
    # print(f"Final model saved to: tennis_ppo_trained_model.zip")

    # # Close the training environment
    # env.close()
    # print("Training environment closed.")

    # # Plotting training results
    # print("\n--- Plotting training results ---")
    # plot_results(LOG_DIR)

    # --- Load and Play with the Trained Model ---
    print("\n--- Starting game with loaded model ---")

    play_env = setup_tennis_environment(render_mode='human')

    # Load the trained model
    try:
        # CORRECTED: Use os.path.join for robust path construction
        # This will correctly handle paths on Windows, Linux, or macOS.
        # If loading the final saved model:
        # MODEL_TO_LOAD_PATH = 'tennis_ppo_trained_model'

        # If loading a specific checkpoint (e.g., best_model_290000):
        MODEL_TO_LOAD_PATH = os.path.join(CHECKPOINT_DIR, 'best_model_290000')  # Example path
        loaded_model = PPO.load(MODEL_TO_LOAD_PATH, device='cuda')
        print(f"Model '{MODEL_TO_LOAD_PATH}.zip' loaded successfully.")
    except FileNotFoundError:
        print("Model not found. Please ensure the model file exists in the specified path.")
        print(f"Attempted to load: {MODEL_TO_LOAD_PATH}.zip")
        play_env.close()
        return

    # --- Playback loop ---
    num_eval_episodes = 5  # Number of episodes to play for evaluation
    MAX_EVAL_STEPS_PER_EPISODE = 2000

    for i in range(num_eval_episodes):
        # Reset the playing environment to start a new episode
        obs = play_env.reset()
        episode_reward = 0
        episode_steps = 0

        # Loop through the game until the episode is terminated, truncated, or max steps reached
        while True:
            # Predict the action. Use deterministic=True for consistent playback.
            action, _states = loaded_model.predict(obs, deterministic=True)

            # Take the action and observe.
            next_obs, reward_array, done_array, info_list = play_env.step(action)

            # For a single environment in DummyVecEnv, we access the first element
            reward = reward_array[0]
            terminated = info_list[0].get('terminated', False)
            truncated = info_list[0].get('truncated', False)

            obs = next_obs  # Update current observation
            episode_reward += reward
            episode_steps += 1

            # Render the game window (only if render_mode='human')
            play_env.render()

            # Check if the episode has ended or max steps reached
            if terminated or truncated or episode_steps >= MAX_EVAL_STEPS_PER_EPISODE:
                print(
                    f"Evaluation Episode {i + 1}/{num_eval_episodes} finished. "
                    f"Total Steps: {episode_steps}, Total Reward: {episode_reward:.2f}. "
                    f"Reason: {'Terminated' if terminated else ('Truncated' if truncated else 'Max Steps')}"
                )
                break  # Break out of the loop for the current episode

    # Close the playing environment after use
    play_env.close()
    print("Evaluation finished.")


if __name__ == "__main__":
    main()
