import gym
import logging
import argparse
import numpy as np

debug = False

parser = argparse.ArgumentParser()

parser.add_argument('--random-exec', action='store_true')

args = parser.parse_args()

if args.random_exec:
    print('Executing gym environment using random actions')
else:
    print('Executing gym environment in Interactive mode')

logging.basicConfig(
    format='%(levelname)s:%(asctime)s:%(message)s', datefmt='%m/%d/%Y %I:%M:%S %p', 
    filename='logs/gamePython.log', encoding='utf-8', level=(logging.DEBUG if debug else logging.INFO)
)

gym_environment = 'Taxi-v3'
env = gym.make(gym_environment)
logging.info('{} GYM environment initiated'.format(gym_environment))

game_count = 1
max_steps = np.Inf 
logging.info('{} Games played with max steps are {}'.format(game_count, max_steps))

episode_reward = 0

observation = env.reset()

while True:
    env.render()

    if args.random_exec:
        action = env.action_space.sample()
    else:
        action = int(input('Specify Action: '))     # nosec

    observation, reward, done, info = env.step(action)
    episode_reward += reward

    print(f'Observation: {observation}; Reward: {reward}; Done: {done}')

    if done:
        break

print(f'Episode Reward: {episode_reward}')
env.close()
