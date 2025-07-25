# Imports
import io
import os
import glob
import torch
import time
import base64
import minigrid
import IPython.display

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Set backend before importing pyplot
import matplotlib.pyplot as plt

import sys
import gymnasium

import stable_baselines3
from stable_baselines3 import DQN
from stable_baselines3.common.results_plotter import ts2xy, load_results
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.env_util import make_atari_env

import mars_lander

import gymnasium as gym
from gymnasium import spaces
from gymnasium.envs.box2d.lunar_lander import *
from gymnasium.wrappers import RecordVideo
from ale_py import ALEInterface


import warnings
warnings.filterwarnings('ignore')

ale = ALEInterface()

AVAILABLE_ENVS = {
    '0': 'LunarLander-v3',
    '1': 'MarsLander-v0'
}

#Let the user choose between environments
print("Available environments:")
for key, env_name in AVAILABLE_ENVS.items():
    print(f"{key}: {env_name}")
selected_key = input("Select environment (0-2): ")
env_name = AVAILABLE_ENVS.get(selected_key, AVAILABLE_ENVS['0'])  # Default to LunarLander-v3


# @title Play Video function
from IPython.display import HTML
from base64 import b64encode
from pyvirtualdisplay import Display

# create the directory to store the video(s)
os.makedirs("./video", exist_ok=True)

display = Display(visible=False, size=(1400, 900))
_ = display.start()

"""
Utility functions to enable video recording of gym environment
and displaying it.
To enable video, just do "env = wrap_env(env)""
"""
def render_mp4(videopath: str) -> str:
  """
  Gets a string containing a b4-encoded version of the MP4 video
  at the specified path.
  """
  mp4 = open(videopath, 'rb').read()
  base64_encoded_mp4 = b64encode(mp4).decode()
  return f'<video width=400 controls><source src="data:video/mp4;' \
         f'base64,{base64_encoded_mp4}" type="video/mp4"></video>'

nn_layers = [64, 64]  # This is the configuration of your neural network. Currently, we have two layers, each consisting of 64 neurons.
                      # If you want three layers with 64 neurons each, set the value to [64,64,64] and so on.

learning_rate = 0.001  # This is the step-size with which the gradient descent is carried out.
                       # Tip: Use smaller step-sizes for larger networks.


log_dir = "/tmp/gym/"
os.makedirs(log_dir, exist_ok=True)

# Create environment
env = gym.make(env_name)
# You can also load other environments like cartpole, MountainCar, Acrobot.
# Refer to https://gym.openai.com/docs/ for descriptions.

# For example, if you would like to load Cartpole,
# just replace the above statement with "env = gym.make('CartPole-v1')".
print('State shape: ', env.observation_space.shape)
print('Number of actions: ', env.action_space.n)

env = stable_baselines3.common.monitor.Monitor(env, log_dir )

callback = EvalCallback(env, log_path=log_dir, deterministic=True)  # For evaluating the performance of the agent periodically and logging the results.
policy_kwargs = dict(activation_fn=torch.nn.ReLU,
                     net_arch=nn_layers)
model = DQN("MlpPolicy", env,policy_kwargs = policy_kwargs,
            learning_rate=learning_rate,
            batch_size=64,  # for simplicity, we are not doing batch update.
            buffer_size=50000,  # size of experience of replay buffer. Set to 1 as batch update is not done
            learning_starts=10000,  # learning starts immediately!
            gamma=0.99,  # discount facto. range is between 0 and 1.
            tau = 0.005,  # the soft update coefficient for updating the target network
            target_update_interval=500,  # update the target network immediately.
            train_freq=(4,"step"),  # train the network at every step.
            max_grad_norm = 10,  # the maximum value for the gradient clipping
            exploration_initial_eps = 1,  # initial value of random action probability
            exploration_fraction = 0.3,  # fraction of entire training period over which the exploration rate is reduced
            gradient_steps = 5,  # number of gradient steps
            seed = 1,  # seed for the pseudo random generators
            verbose=1)  # Set verbose to 1 to observe training logs. We encourage you to set the verbose to 1.

# You can also experiment with other RL algorithms like A2C, PPO, DDPG etc.
# Refer to  https://stable-baselines3.readthedocs.io/en/master/guide/examples.html
# for documentation. For example, if you would like to run DDPG, just replace "DQN" above with "DDPG".


#training the DQN Model
model.learn(total_timesteps=10000, log_interval=10, callback=callback)
# The performance of the training will be printed every 10000 episodes. Change it to 1, if you wish to
# view the performance at every training episode.
##print("Model saved as lunar_lander_dqn.zip")

timestamp = int(time.time())
env = gym.make(env_name, render_mode="rgb_array")
env = gym.wrappers.RecordVideo(
    env,
    video_folder="video",
    name_prefix=f"{env_name}_learned_{timestamp}",
    episode_trigger=lambda episode_id: True
)

observation, _ = env.reset()
total_reward = 0
done = False

while not done:
    action, states = model.predict(observation, deterministic=True)
    observation, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
    total_reward += reward

env.close()
print(f"\nTotal reward: {total_reward}")

# show video, doesn't work currently
#html = render_mp4(f"video/{env_name}_learned-episode-0.mp4")
#HTML(html)

x, y = ts2xy(load_results(log_dir), 'timesteps')  # Organising the logged results in to a clean format for plotting.
plt.plot(x, y)
plt.ylim([-300, 300])
plt.xlabel('Timesteps')
plt.ylabel('Episode Rewards')
plt.savefig('training_plot.png')  # Save the plot
print("Plot saved as training_plot.png. Check the file to verify.")
plt.close()  # Clean up