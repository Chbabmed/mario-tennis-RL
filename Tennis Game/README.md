# Tennis RL Agent (PPO with Custom Rewards) - Libraries and Resources

## Dependencies

Before running the code, ensure you have the following Python libraries installed:

* `Python 3.8+`

* `gymnasium`

* `ale-py` (required by `gymnasium` for Atari environments)

* `numpy`

* `matplotlib`

* `stable-baselines3` (with `[extra]` for TensorBoard, `pip install stable-baselines3[extra]`)

* `tensorflow` or `pytorch` (Stable Baselines3 uses one of these backends; `tensorflow` is indicated in your logs)

You can install most of these using pip:

```bash
pip install gymnasium ale-py numpy matplotlib stable-baselines3[extra]
```

## References

* **Gymnasium Documentation:**
    * <https://gymnasium.farama.org/>
    * [Wrappers: https://gymnasium.farama.org/api/wrappers/](https://gymnasium.farama.org/api/wrappers/)
* **Stable Baselines3 Documentation:**
    * <https://stable-baselines3.readthedocs.io/>
    * [PPO Algorithm: https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html](https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html)
    * [Custom Environments and Wrappers: https://stable-baselines3.readthedocs.io/en/master/guide/custom_env.html](https://www.google.com/search?q=https://stable_baselines3.readthedocs.io/en/master/guide/custom_env.html)
* **Arcade Learning Environment (ALE):**
    * [https://www.arcadelearningenvironment.org/](https://www.google.com/search?q=https://www.arcadelearningenvironment.org/)
* **Proximal Policy Optimization (PPO) Paper:**
    * "Proximal Policy Optimization Algorithms" by John Schulman et al. (2017)
        * <https://arxiv.org/abs/1707.06347>
* **Reward Shaping:**
    * "Reward Shaping" on Wikipedia: [https://en.wikipedia.org/wiki/Reward_shaping](https://www.google.com/search?q=https://en.wikipedia.org/wiki/Reward_shaping)
    * "Potential-Based Reward Shaping" by Andrew Y. Ng et al. (1999): A foundational paper on a specific type of reward shaping.
        * [https://web.stanford.edu/~ang/papers/shaping-nips99.pdf](https://www.google.com/search?q=https://web.stanford.edu/~ang/papers/shaping-nips99.pdf)
