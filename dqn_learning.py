import gym
import tensorflow as tf
import numpy as np
import keras

import util
from q_table_learning import QTable_Learning
from gym_environment import GymEnvironment


class DQN_Learning(QTable_Learning):
    def __init__(
        self,
        train_env: GymEnvironment,
        test_env: GymEnvironment,
        alpha: float = 1,
        epsilon: float = 0.05,
        gamma: float = 0.8,
        model_file: str = None,
        learning_rate: float = 1e-5,
        initial_state_collection: int = 1e5
    ) -> None:
        super().__init__(train_env, alpha, epsilon, gamma, model_file)
        self.model_file = "dqnModel"
        self.dqn_learning_rate = learning_rate
        self.dqn_model = self.create_model()
        self.buffer = util.ReplayBuffer(10000000, 1)
        self.evaluation_env = test_env
        self.initial_state_collection = initial_state_collection

    def create_model(self, hidden_layer_units: list = [100, 50]):
        model = keras.Sequential(name="DQNModel")
        model.add(tf.keras.Input(shape=(4,)))
        for i, units in enumerate(hidden_layer_units):
            model.add(tf.keras.layers.Dense(
                units,
                activation=tf.keras.activations.relu,
                kernel_initializer=tf.keras.initializers.VarianceScaling(
                    scale=2.0,
                    mode='fan_in',
                    distribution='truncated_normal'
                ), name=f"HiddenLayer{i}"
            ))
        model.add(tf.keras.layers.Dense(
            6,
            activation=None,
            kernel_initializer=tf.keras.initializers.RandomUniform(
                minval=-0.03, maxval=0.03),
            bias_initializer=tf.keras.initializers.Constant(-0.2), name="outputLayer"
        ))

        model.compile(
            optimizer=tf.keras.optimizers.Adam(
                learning_rate=self.dqn_learning_rate),
            loss=tf.keras.losses.MeanSquaredError(),
            metrics=[tf.keras.metrics.MeanSquaredError()]
        )

        model.summary()

        return model

    def compute_action(self, observation: np.ndarray):
        predicted_qvalue = self.model.predict(np.array(observation))
        return predicted_qvalue[0].argmax()

    def random_action(self, observation: np.ndarray):
        if util.flipCoin(self.epsilon):
            return self.env.sample_action()
        return self.compute_action(observation)

    def q_value_calculation(self, state: np.ndarray, next_state: np.ndarray, reward: int, action: int):
        state_qvalue = max(self.model.predict(np.array([state, next_state])))
        sample = self.alpha * (reward + (self.gamma * max(state_qvalue[1])))
        state_action_qvalue = (1 - self.alpha) * state_qvalue[0][action]
        return state_action_qvalue + sample

    def collect_data(self, env: gym.Env, observation: np.ndarray, size: int):
        for _ in range(size):
            idx = self.buffer.store_frame(observation)
            action = self.random_action(observation=observation)
            observation, reward, done, _ = env.step(action)
            self.buffer.store_effect(idx, action, reward, done)
            if done:
                observation = env.reset()

    def batch_creation(self, batch_size: int):
        if not self.buffer.can_sample(batch_size=batch_size):
            raise ValueError(
                f'The buffer does not have adequate amount of states; current state count: {self.buffer.num_in_buffer}')
        state_batch, action_batch, reward_batch, new_state_batch = self.buffer.sample(
            batch_size=batch_size)
        q_value_batch = np.array([
            self.q_value_calculation(state, next_state, reward, action)
            for state, next_state, reward, action in zip(state_batch, action_batch, reward_batch, new_state_batch)
        ])
        return state_batch, q_value_batch

    def train_model(self, iteration: int, epoch: int, batch: int, add_to_buffer: int):
        obs = self.env.env.reset()
        self.collect_data(env=self.env.env, observation=obs,
                          size=self.initial_state_collection)
        observation_batch, q_value_batch = self.batch_creation(
            batch_size=batch)
        
        for _ in range(iteration):

            self.model.fit(observation_batch, q_value_batch,
                           batch_size=batch, epochs=epoch)

            self.collect_data(env=self.env.env,
                              observation=obs, size=add_to_buffer)
            observation_batch, q_value_batch = self.batch_creation(
                batch_size=batch)

    def evaluate_model(self, episode_count: int):
        observation = self.evaluation_env.env.reset()
        evaluation_ep_reward = []
        evaluation_ep_step = []
        for _ in range(episode_count):
            episode_step_count = 0
            episode_reward = 0
            while True:
                action = self.compute_action(observation=observation)
                observation, reward, done, _ = self.evaluation_env.env.step(
                    action)
                episode_reward += reward
                episode_step_count += 1
                if done:
                    break
            evaluation_ep_reward.append(episode_reward)
            evaluation_ep_step.append(episode_step_count)
