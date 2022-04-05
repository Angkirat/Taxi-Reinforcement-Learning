from GymEnvironment import GymEnvironment
from statistics import mean
from tqdm import tqdm
import logging
import util
import pickle   # nosec
import os
import tensorflow as tf


class QTable_Learning:

    def __init__(self, env: GymEnvironment, alpha: float = 1.0, epsilon: float = 0.05, gamma: float = 0.8, model_file: str = None) -> None:
        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self._q_val = util.Counter()
        self.model_file = 'ModelOutput/RL-Qlearning.pkl' if model_file is None else model_file
        self.load_model()

    def get_qval(self, state, action):
        return self._q_val[(state, action)]

    def getAction(self, state) -> int:
        if util.flipCoin(self.epsilon):
            return self.env.sample_action()
        return self.getPolicy(state)

    def updateQvalues(self, state: int, action: int, reward: float, nextState: int) -> None:
        sample = self.alpha * \
            (reward + (self.gamma * self.computeStateQvalue(nextState)))
        self._q_val[(state, action)] = ((1 - self.alpha) *
                                        self._q_val[(state, action)]) + sample

    def getPolicy(self, state: int) -> int:
        action_cntr = util.Counter()
        for action in self.env.getPossibleActions():
            action_cntr[action] = self.get_qval(state, action)
        return action_cntr.argMax()

    def store_model(self) -> None:
        folder_path = os.path.dirname(self.model_file)
        if len(folder_path) != 0:
            os.makedirs(folder_path, exist_ok=True)
        pickle.dump(self._q_val, open(self.model_file, 'wb'))

    def load_model(self) -> None:
        if not os.path.isfile(self.model_file):
            logging.info('No model created')
            return None
        self._q_val = pickle.load(open(self.model_file, 'rb'))    # nosec

    def computeStateQvalue(self, state: int) -> float:
        return max([self.get_qval(state, action) for action in self.env.getPossibleActions()])

    def train(self, iteration_count: int, saveAfter: int = 1) -> None:
        print(self.env.envName)
        iteration_count += 1
        for i in tqdm(range(1, iteration_count)):
            episode_reward = 0
            step = 0

            old_observation = 0
            observation = self.env.env.reset()
            while True:
                # self.env.env.render()

                old_observation = observation
                action = self.getAction(observation)
                observation, reward, done, info = self.env.env.step(action)

                self.updateQvalues(old_observation, action,
                                   reward, observation)

                episode_reward += reward
                step += 1

                if (step % 500) == 0:
                    logging.debug(
                        f'Training Iteration {i}; Total Steps {step}; Reward gained: {episode_reward}; Action: {action}; status: {done}')

                if reward == 20:
                    logging.info(
                        f'Training Iteration {i}; total Steps Taken {step}; Rewards gained: {episode_reward}')
                    break

            if (i % saveAfter) == 0:
                logging.info(
                    f'Training Iteration {i}; Creating pickle dump of the Q-values')
                self.store_model()
        self.store_model()

    def evaluate(self, iteration_count: int):
        reward_list = []
        step_list = []

        logging.info("\n\n")
        logging.info("=="*50)
        logging.info("\n\n")

        for i in tqdm(range(iteration_count)):
            episode_reward = 0
            step = 0

            observation = self.env.env.reset()
            while True:
                # self.env.env.render()

                action = self.getPolicy(observation)
                observation, reward, done, _ = self.env.env.step(action)

                episode_reward += reward
                step += 1

                if (step % 500) == 0:
                    logging.debug(
                        f'Testing Iteration {i}; Total Steps {step}; Reward gained: {episode_reward}; Action: {action}; status: {done}')

                if reward == 20:
                    logging.info(
                        f'Testing Iteration {i}; total Steps Taken {step}; Rewards gained: {episode_reward}')
                    break

            reward_list.append(episode_reward)
            step_list.append(step)

        logging.info('=='*50)
        logging.info(
            f'Testing Total Iterations {iteration_count}; Mean Reward {mean(reward_list)}; Averge Steps taken are {mean(step_list)}')
        logging.info('=='*50)

        print('=='*50)
        print('Test Execution - Q Learning')
        print('Mean Reward: {}'.format(mean(reward_list)))
        print('Mean Steps: {}'.format(mean(step_list)))
        print('=='*50)
