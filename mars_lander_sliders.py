import gymnasium as gym
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
import mars_lander  # Import to register MarsLander-v0
import os
import numpy as np
from stable_baselines3.common.results_plotter import ts2xy, load_results


# At the top of param_slider.py, after imports
try:
    import matplotlib
    matplotlib.use('TkAgg')
    import tkinter  # Explicitly import to verify
except ImportError as e:
    print(f"Error: {e}. Install Tkinter with 'sudo apt install python3-tk' and restart.")
    exit(1)
# Function to train and evaluate the model with current params
def train_and_evaluate(gravity, wind_power, timesteps=10000):
    log_dir = "/tmp/gym/"
    os.makedirs(log_dir, exist_ok=True)

    env = gym.make('MarsLander-v0', gravity=gravity, wind_power=wind_power, render_mode='rgb_array')
    env = Monitor(env, log_dir)

    callback = EvalCallback(env, log_path=log_dir, deterministic=True, verbose=0)

    policy_kwargs = dict(net_arch=[64, 64])  # Your network layers

    model = DQN(
        "MlpPolicy",
        env,
        policy_kwargs=policy_kwargs,
        learning_rate=0.001,
        batch_size=64,
        buffer_size=50000,
        learning_starts=1000,
        gamma=0.99,
        tau=0.005,
        target_update_interval=500,
        train_freq=(4, "step"),
        gradient_steps=5,
        exploration_initial_eps=1.0,
        exploration_fraction=0.3,
        verbose=1,  # Silent training
    )

    # Train for 10k steps (short for Pi)
    model.learn(total_timesteps=timesteps, callback=callback)

    # Evaluate: Run 10 episodes
    rewards = []
    for _ in range(10):
        obs, _ = env.reset()
        done = False
        episode_reward = 0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            episode_reward += reward
            done = terminated or truncated
        rewards.append(episode_reward)

    mean_reward = np.mean(rewards)
    env.close()

    x, y = ts2xy(load_results(log_dir), 'timesteps')
    return x, y, mean_reward

# Initial params (Mars defaults)
initial_gravity = -3.71
initial_wind = 10.0

# Create the plot and sliders
fig, (ax_plot, ax_info) = plt.subplots(2, 1, figsize=(8, 6))
plt.subplots_adjust(left=0.25, bottom=0.3)

# Initial training run for baseline
x, y, mean_reward = train_and_evaluate(initial_gravity, initial_wind)
line, = ax_plot.plot(x, y)
ax_plot.set_title("Training Reward Curve (Gravity: {:.2f}, Wind: {:.2f})".format(initial_gravity, initial_wind))
ax_plot.set_xlabel('Timesteps')
ax_plot.set_ylabel('Episode Rewards')
ax_plot.set_ylim([-300, 300])

ax_info.text(0.5, 0.5, f"Mean Reward (10 episodes): {mean_reward:.2f}", ha='center', va='center')
ax_info.axis('off')

# Gravity slider
ax_gravity = plt.axes([0.25, 0.15, 0.65, 0.03])
gravity_slider = Slider(ax_gravity, 'Gravity', -12.0, 0.0, valinit=initial_gravity, valstep=0.1)

# Wind Power slider
ax_wind = plt.axes([0.25, 0.1, 0.65, 0.03])
wind_slider = Slider(ax_wind, 'Wind Power', 0.0, 20.0, valinit=initial_wind, valstep=1.0)

# Train button
ax_button = plt.axes([0.8, 0.025, 0.1, 0.04])
button = Button(ax_button, 'Train')

# Update function for button press
def update(event):
    gravity = gravity_slider.val
    wind = wind_slider.val
    x, y, mean_reward = train_and_evaluate(gravity, wind, timesteps=50000)
    line.set_xdata(x)
    line.set_ydata(y)
    ax_plot.set_title("Training Reward Curve (Gravity: {:.2f}, Wind: {:.2f})".format(gravity, wind))
    ax_plot.relim()
    ax_plot.autoscale_view()
    ax_info.cla()
    ax_info.text(0.5, 0.5, f"Mean Reward (10 episodes): {mean_reward:.2f}", ha='center', va='center')
    ax_info.axis('off')
    fig.canvas.draw_idle()

button.on_clicked(update)

plt.show()