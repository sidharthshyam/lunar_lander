#!/usr/bin/env python3
import os
import itertools
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor

# 1) Hyperparameter grid
learning_rates      = [0.001]
exploration_fracs   = [0.5, 0.3]
nn_layer_configs    = [ [128,128], [64,64,64], [64,64]]

# 2) Paths
LOG_DIR    = "./tmp_gym/"
os.makedirs(LOG_DIR, exist_ok=True)
RESULT_CSV = "hyperparam_tuning_results.csv"

# 3) Init results CSV
pd.DataFrame(
    columns=[
        "learning_rate","exploration_fraction","nn_layers",
        "mean_reward","mean_ep_length","timesteps"
    ]
).to_csv(RESULT_CSV, index=False)

# 4) Loop over each combo
for lr, ef, layers in itertools.product(learning_rates, exploration_fracs, nn_layer_configs):
    # a) create fresh env + monitor
    env = gym.make("LunarLander-v3")
    env = Monitor(env, LOG_DIR)
    
    # b) evaluation callback (optional saving best model)
    callback = EvalCallback(
        env,
        eval_freq=10000,
        best_model_save_path=LOG_DIR,
        log_path=LOG_DIR,
        deterministic=True,
        verbose=0
    )
    
    # c) build & train
    policy_kwargs = dict(activation_fn=torch.nn.ReLU, net_arch=layers)
    model = DQN(
        "MlpPolicy", env,
        learning_rate=lr,
        exploration_initial_eps=1.0,
        exploration_fraction=ef,
        buffer_size=50000,
        batch_size=64,
        train_freq=(4, "step"),
        gradient_steps=4,
        target_update_interval=500,
        policy_kwargs=policy_kwargs,
        verbose=1,
    )
    TIMESTEPS = 100000
    model.learn(total_timesteps=TIMESTEPS, callback=callback)
    
    # d) read the actual monitor.csv (skip any comments)
    monitor_path = os.path.join(LOG_DIR, "monitor.csv")
    df = pd.read_csv(monitor_path, comment='#')
    mean_r   = df["r"].mean()
    mean_len = df["l"].mean()
    
    # e) append to results CSV
    pd.DataFrame([{
        "learning_rate":      lr,
        "exploration_fraction": ef,
        "nn_layers":          str(layers),
        "mean_reward":        mean_r,
        "mean_ep_length":     mean_len,
        "timesteps":          TIMESTEPS
    }]).to_csv(RESULT_CSV, mode="a", header=False, index=False)
    
    # f) clean up before next run
    for f in os.listdir(LOG_DIR):
        os.remove(os.path.join(LOG_DIR, f))

# 5) Plot heatmap of mean rewards
pivot = df_res.pivot_table(
    index="learning_rate",
    columns="exploration_fraction",
    values="mean_reward"
)

plt.figure(figsize=(6,5))
plt.title("Mean Reward Heatmap")
plt.xlabel("Exploration Fraction")
plt.ylabel("Learning Rate")
plt.imshow(pivot, origin="lower", aspect="auto", cmap="viridis")
plt.colorbar(label="Mean Reward")
plt.xticks(range(len(pivot.columns)), pivot.columns)
plt.yticks(range(len(pivot.index)), pivot.index)
plt.tight_layout()
plt.savefig("hyperparam_reward_heatmap.png")
print("▶ Hyperparameter tuning complete.")
print(f"  • Results: {RESULT_CSV}")
print("  • Heatmap: hyperparam_reward_heatmap.png")