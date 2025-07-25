import gymnasium as gym
import pygame
import mars_lander  # Import to register MarsLander-v0
from stable_baselines3 import DQN
from gymnasium import spaces
from gymnasium.envs.box2d.lunar_lander import *
from ale_py import ALEInterface
from concurrent.futures import ThreadPoolExecutor  # For parallel runs

AVAILABLE_ENVS = {
    '1': 'LunarLander-v3',
    '2': 'MarsLander-v0'
}

#Let the user choose between environments
print("Available environments:")
for key, env_name in AVAILABLE_ENVS.items():
    print(f"{key}: {env_name}")
selected_key = input("Select environment (1-2): ")
env_name = AVAILABLE_ENVS.get(selected_key, AVAILABLE_ENVS['1'])  # Default to LunarLander-v3


# Load AI model
model = DQN.load("lunar_lander_dqn.zip")

# Human input function (using Pygame for keys)
def human_input(env):
    pygame.init()
    screen = pygame.display.set_mode((600,400)) # Human window
    clock = pygame.time.Clock()
    observation, _ = env.reset()
    done = False
    total_reward = 0

    while not done:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True

        keys = pygame.key.get_pressed()
        action = 0  # Noop
        if keys[pygame.K_LEFT]:
            action = 1  # Left engine
        elif keys[pygame.K_RIGHT]:
            action = 3  # Right engine
        elif keys[pygame.K_DOWN]:
            action = 2  # Main engine

        observation, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        done = terminated or truncated

        # Render human env
        frame = env.render()
        screen.blit(pygame.surfarray.make_surface(frame.swapaxes(0, 1)), (0, 0))
        pygame.display.flip()
        clock.tick(50)  # Match FPS

    env.close()
    pygame.quit()
    return total_reward

# AI run function
def ai_run(env):
    observation, _ = env.reset()
    done = False
    total_reward = 0

    while not done:
        action, _ = model.predict(observation, deterministic=True)
        observation, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        done = terminated or truncated

        # Optional: Render AI env in separate window if desired

    env.close()
    return total_reward

# Run human vs AI in parallel
with ThreadPoolExecutor() as executor:
    human_env = gym.make(env_name, render_mode='rgb_array')
    ai_env = gym.make(env_name, render_mode='rgb_array')  # No render for AI, or 'human' for second window

    human_future = executor.submit(human_input, human_env)
    ai_future = executor.submit(ai_run, ai_env)

    human_reward = human_future.result()
    ai_reward = ai_future.result()

print(f"Human Reward: {human_reward}")
print(f"AI Reward: {ai_reward}")
print("Human wins!" if human_reward > ai_reward else "AI wins!" if ai_reward > human_reward else "Tie!")