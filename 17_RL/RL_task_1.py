# %%
# Import env
import gym
import numpy as np
from time import sleep

# Create carpole environment
env = gym.make("CartPole-v1")


# %%
# Loop over
env.reset()
env.render()
reward_sum = 0
terminated = False

while not terminated:
    action = np.random.randint(0, 2)
    s_prime, reward, terminated, info = env.step(action)
    env.render()
    print(reward, terminated)
    reward_sum += reward
    sleep(0.5)

print("total reward: ", reward_sum)
print("Good by!")


# %%
