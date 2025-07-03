Tennis RL Agent (PPO with Custom Rewards)
This project implements a Reinforcement Learning agent to play the Atari 2600 game "Tennis" (ALE/Tennis-v5) using the Proximal Policy Optimization (PPO) algorithm. A key feature of this project is the integration of a custom reward shaping mechanism to guide the agent's learning, particularly for events like hitting the ball.

Table of Contents
Project Overview

Features

Environment

Dependencies

Installation

Usage

1. Training the Agent

2. Playing with a Trained Model

3. Inspecting Pixel Values for Reward Shaping

Custom Reward System (Reward Shaping)

Enhancing Learning (Hyperparameters & Strategies)

References

Project Overview
This repository contains Python code for training an AI agent to play the classic Atari 2600 game "Tennis". It leverages the gymnasium library for environment interaction and stable-baselines3 for the PPO reinforcement learning algorithm. A custom CustomRewardWrapper is implemented to provide additional, fine-grained reward signals to the agent, aiming to accelerate learning beyond the sparse default game rewards (only +1 for winning a point, -1 for losing a point).

Features
Proximal Policy Optimization (PPO): Utilizes the PPO algorithm from stable-baselines3 for robust and efficient policy learning.

Pixel-Based Observations: Processes raw game pixels (converted to grayscale and stacked frames) as observations for the agent's convolutional neural network (CNN) policy.

Custom Reward Shaping: Implements a CustomRewardWrapper to:

Amplify existing game rewards (make wins/losses more impactful).

Penalize inactivity (NOOP actions).

Provide a small reward for hitting the ball (detected via pixel analysis).

Penalize the player for moving horizontally out of the defined playing field.

Training Checkpoints: Automatically saves model checkpoints periodically during training.

Training Progress Plotting: Generates a plot of episode rewards over time to visualize learning progress.

Evaluation Mode: Allows loading a pre-trained model to observe its performance in real-time with rendering.

Environment
The project uses the ALE/Tennis-v5 environment from the Arcade Learning Environment (ALE) via gymnasium.

Observation Space: Grayscale pixel frames (84x84) stacked 4 times.

Action Space: Discrete actions representing joystick movements and the "fire" button.

Dependencies
Before running the code, ensure you have the following Python libraries installed:

Python 3.8+

gymnasium

ale-py (required by gymnasium for Atari environments)

numpy

matplotlib

stable-baselines3 (with [extra] for TensorBoard, pip install stable-baselines3[extra])

tensorflow or pytorch (Stable Baselines3 uses one of these backends; tensorflow is indicated in your logs)

You can install most of these using pip:

pip install gymnasium ale-py numpy matplotlib stable-baselines3[extra]

For GPU support (device='cuda'), ensure you have a compatible NVIDIA GPU and the correct versions of CUDA Toolkit and cuDNN installed, along with the GPU-enabled versions of TensorFlow or PyTorch.

Installation
Clone the repository (or save the provided Python code as a .py file, e.g., tennis_agent.py).

Install dependencies as listed above.

Usage
1. Training the Agent
To train the agent, ensure the training section in the main() function is uncommented.

# --- 5. Main Function for Training and Evaluation ---
def main():
    # ... (directory setup) ...

    # --- Setup Environment for Training ---
    env = setup_tennis_environment(render_mode=None, log_dir=LOG_DIR)
    print("Training environment setup complete.")

    callback = TrainAndLoggingCallback(check_freq=10000, save_path=CHECKPOINT_DIR)
    # ... (PPO model instantiation) ...

    # --- Train the RL Model ---
    TRAINING_TIMESTEPS = 500000
    print(f"Starting model training for {TRAINING_TIMESTEPS} timesteps...")
    model.learn(total_timesteps=TRAINING_TIMESTEPS, callback=callback)
    print("Model training finished.")

    # ... (save final model, close env, plot results) ...

    # --- Load and Play with the Trained Model (COMMENTED OUT if only training) ---
    # ...

Run the script from your terminal:

python tennis_agent.py

Training logs will be saved in the ./logs directory (viewable with TensorBoard).

Model checkpoints (best_model_X.zip) will be saved in the ./checkpoints directory.

A final model (tennis_ppo_trained_model.zip) will be saved at the end of training.

A plot of training rewards will be displayed upon completion.

2. Playing with a Trained Model
To play with a previously trained model without retraining, ensure the training section in main() is commented out, and the "Load and Play" section is uncommented.

# --- 5. Main Function for Training and Evaluation ---
def main():
    # ... (directory setup) ...

    # --- Setup Environment for Training (COMMENTED OUT) ---
    # ... (all training related code commented out) ...

    # --- Load and Play with the Trained Model ---
    print("\n--- Starting game with loaded model ---")

    play_env = setup_tennis_environment(render_mode='human')

    # Load the trained model
    try:
        # Example: To load a specific checkpoint (e.g., from 290,000 timesteps)
        MODEL_TO_LOAD_PATH = os.path.join(CHECKPOINT_DIR, 'best_model_290000') 
        # Or to load the final saved model:
        # MODEL_TO_LOAD_PATH = 'tennis_ppo_trained_model'

        loaded_model = PPO.load(MODEL_TO_LOAD_PATH, device='cuda')
        print(f"Model '{MODEL_TO_LOAD_PATH}.zip' loaded successfully.")
    except FileNotFoundError:
        print("Model not found. Please ensure the model file exists in the specified path.")
        print(f"Attempted to load: {MODEL_TO_LOAD_PATH}.zip")
        play_env.close()
        return

    # ... (playback loop) ...

Important: Update MODEL_TO_LOAD_PATH to point to the exact .zip file of the model you wish to load (e.g., './checkpoints/best_model_290000' or 'tennis_ppo_trained_model').

Run the script:

python tennis_agent.py

A game window will open, and the loaded agent will play for num_eval_episodes (default 5) episodes, each limited by MAX_EVAL_STEPS_PER_EPISODE (default 2000 steps).

3. Inspecting Pixel Values for Reward Shaping
To fine-tune the BALL_COLOR_LOW, BALL_COLOR_HIGH, PLAYER_PADDLE_COLOR_LOW, PLAYER_PADDLE_COLOR_HIGH, PLAYER_PADDLE_Y_RANGE, and BALL_Y_RANGE values in CustomRewardWrapper, you can use a temporary script to visualize a grayscale frame and inspect its pixel values:

import gymnasium as gym
import ale_py
import numpy as np
import matplotlib.pyplot as plt
from gymnasium.wrappers import GrayScaleObservation

def get_single_grayscale_frame():
    gym.register_envs(ale_py)
    env = gym.make('ALE/Tennis-v5', render_mode='rgb_array')
    env = GrayScaleObservation(env, keep_dim=True)
    obs, _ = env.reset()
    for _ in range(50): # Take a few steps to get a dynamic frame
        action = env.action_space.sample()
        obs, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            obs, _ = env.reset()
    env.close()
    return np.squeeze(obs)

if __name__ == "__main__":
    grayscale_frame = get_single_grayscale_frame()
    plt.figure(figsize=(8, 6))
    plt.imshow(grayscale_frame, cmap='gray')
    plt.title('Sample Grayscale Frame from Tennis')
    plt.colorbar(label='Pixel Value (0-255)')
    plt.show()
    print("Hover your mouse over the plot to see pixel values and determine thresholds.")

Run this script, hover your mouse over the ball and paddles in the displayed image, and note the pixel values to adjust the thresholds in CustomRewardWrapper.

Custom Reward System (Reward Shaping)
The CustomRewardWrapper modifies the default reward signal from the environment to provide more nuanced feedback to the agent.

Current Reward Shaping Logic:

Amplify Game Rewards:

+0.1 bonus for winning a point (on top of the default +1).

-0.1 penalty for losing a point (on top of the default -1).

Penalize Inactivity:

-0.0001 penalty for taking the NOOP (Action 0) action. This encourages the agent to be more active.

Reward Ball Hits:

+0.005 bonus for detecting the player's paddle hitting the ball. This is a heuristic based on pixel proximity:

Ball Color Range: Pixels with intensity 240-255 (very bright white).

Player Paddle Color Range: Pixels with intensity 170-200 (light gray).

Player Paddle Y-Range: (120, 180) (vertical area where the player's paddle is expected).

Ball Y-Range: (50, 180) (vertical area where the ball is expected to be in play).

Hit Distance Threshold: If the center of the detected ball and paddle are within 8 pixels.

Penalize Out-of-Field Movement:

-0.01 penalty if the player's paddle moves horizontally outside the defined playing boundaries (FIELD_X_MIN = 10, FIELD_X_MAX = 150).

These values are tunable and may require further experimentation to optimize agent performance.

Enhancing Learning (Hyperparameters & Strategies)
To further improve the agent's performance, consider exploring these areas:

Hyperparameter Tuning: Experiment with the PPO algorithm's parameters in the model = PPO(...) instantiation, such as:

learning_rate: Adjust from 1e-6 (current) to 1e-5 or 3e-4.

n_steps: Number of steps collected per environment before an update (e.g., 128, 2048).

batch_size: Size of minibatches for optimization (e.g., 64, 256).

n_epochs: Number of times the collected data is iterated over for updates.

gamma: Discount factor (closer to 1 for long-term rewards).

ent_coef: Entropy coefficient (encourages exploration).

clip_range: PPO clipping parameter.

Increased Training Timesteps: Atari games often require millions (e.g., 10M, 20M) of timesteps for robust performance. Your current 500,000 is a good start but might not be sufficient for optimal play.

Learning Rate Schedule: Implement a decaying learning rate (e.g., linear decay) to allow for faster initial learning and finer tuning later.

Observation Normalization: While CnnPolicy handles some internal scaling, explicit VecNormalize can sometimes help, especially if you also normalize rewards. Remember to save and load the VecNormalize object.

More Advanced Reward Shaping: If pixel-based detection proves unreliable, consider exploring techniques to access Atari 2600 RAM values if you need precise game state information (e.g., exact ball coordinates, scores, game phase). This is a more advanced topic.

References
Gymnasium Documentation:

https://gymnasium.farama.org/

Wrappers: https://gymnasium.farama.org/api/wrappers/

Stable Baselines3 Documentation:

https://stable-baselines3.readthedocs.io/

PPO Algorithm: https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html

Custom Environments and Wrappers: https://stable-baselines3.readthedocs.io/en/master/guide/custom_env.html

Arcade Learning Environment (ALE):

https://www.arcadelearningenvironment.org/

Proximal Policy Optimization (PPO) Paper:

"Proximal Policy Optimization Algorithms" by John Schulman et al. (2017)

https://arxiv.org/abs/1707.06347

Reward Shaping:

"Reward Shaping" on Wikipedia: https://en.wikipedia.org/wiki/Reward_shaping

"Potential-Based Reward Shaping" by Andrew Y. Ng et al. (1999): A foundational paper on a specific type of reward shaping.

https://web.stanford.edu/~ang/papers/shaping-nips99.pdf
