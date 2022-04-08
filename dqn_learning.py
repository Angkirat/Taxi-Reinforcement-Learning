import gym
import pickle   #nosec
import logging
import numpy as np
from tqdm import tqdm
import tensorflow as tf
from statistics import mean
from datetime import datetime

import util
from gym_environment import GymEnvironment
from Modified_Taxi_Environment import ENV_NAME, registerEnvironment

logdir = "logs/DQN/" + datetime.now().strftime("%Y%m%d-%H%M%S")
file_writer = tf.summary.create_file_writer(logdir + "/metrics")
file_writer.set_as_default()
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)

logging.basicConfig(
    format='%(levelname)s:%(asctime)s:%(message)s', datefmt='%m/%d/%Y %I:%M:%S %p',
    filename='logs/DQN-Learning.log', encoding='utf-8', level=(logging.DEBUG)
)

class DQN_Learning:
    def __init__(
        self,
        env: GymEnvironment, alpha: float = 1,
        epsilon: float = 0.05, gamma: float = 0.8,
        learning_rate: float = 1e-5,
        hidden_layer_units: list = [100, 50], dropout: list = [0.2, 0.2],
        initial_state_collection: int = 1e5, model_save_path: str = 'ModelOutput/DQN_leanring'
    ) -> None:
        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.dqn_learning_rate = learning_rate
        self.model = None
        self.buffer = util.ReplayBuffer(1e10, 1)
        self.evaluation_env = env
        self.initial_state_collection = initial_state_collection
        self.hidden_layer_units = hidden_layer_units
        self.dropout = dropout
        self.model_save_path = model_save_path

    def training_main(self, training_episode_count: int, test_episode_count: int):
        self.model = self.create_model()
        training_loss, evaluation_metrics = self.train_model_episode(
            training_episode_count,
            test_episode_count
        )
        evaluation_ep_reward, evaluation_ep_step = self.evaluate_model(200)
        summary_model = {
            'training_loss': training_loss,
            'evaluation_metrics': evaluation_metrics,
            'evaluation_ep_reward': evaluation_ep_reward,
            'evaluation_ep_step': evaluation_ep_step
        }
        pickle.dump(summary_model, open('ModelOutput/DQN-Metrics_1.pkl', 'wb'))


    def create_model(self):
        model = tf.keras.Sequential(name="DQNModel")
        model.add(tf.keras.Input(shape=self.env.obs_shape))
        for i, (units, drop) in enumerate(zip(self.hidden_layer_units, self.dropout)):
            model.add(tf.keras.layers.Dense(
                units,
                activation=tf.keras.activations.relu,
                kernel_initializer=tf.keras.initializers.VarianceScaling(
                    scale=2.0,
                    mode='fan_in',
                    distribution='truncated_normal'
                ), name=f"Hidden_Layer_{(i+1)}"
            ))
            model.add(tf.keras.layers.Dropout(
                drop,
                name=f"Dropout_Layer_{(i+1)}"
            ))
        model.add(tf.keras.layers.Dense(
            self.env.action_size,
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

        return model

    def compute_action(self, observation: np.ndarray):
        predicted_qvalue = self.model.predict(np.array([observation]))
        return predicted_qvalue[0].argmax()

    def random_action(self, observation: np.ndarray):
        if util.flipCoin(self.epsilon):
            return self.env.sample_action()
        return self.compute_action(observation)

    def q_value_calculation(self, state: np.ndarray, next_state: np.ndarray, reward: int, action: int):
        state_qvalue = self.model.predict(np.array([state, next_state]))
        sample = self.alpha * (reward + (self.gamma * max(state_qvalue[1])))
        state_q_value = state_qvalue[0]
        state_action_qvalue = (1 - self.alpha) * state_q_value[action]
        state_q_value[action] = state_action_qvalue + sample
        return state_q_value

    def collect_data(self, env: gym.Env, observation: np.ndarray, size: int):
        print('\n\n')
        print('Collecting data in the Replay Buffer')

        for _ in tqdm(range(size)):
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
            for state, action, reward, next_state in
            zip(state_batch, action_batch, reward_batch, new_state_batch)
        ])
        return state_batch, q_value_batch

    def train_model(self, iteration: int, epoch: int, batch: int, add_to_buffer: int):
        obs = self.env.env.reset()
        self.collect_data(env=self.env.env, observation=obs,
                          size=self.initial_state_collection)
        observation_batch, q_value_batch = self.batch_creation(
            batch_size=batch)

        print('\n\n')
        print('Training DQN model (random state)')

        for _ in tqdm(range(iteration)):

            self.model.fit(observation_batch, q_value_batch,
                           batch_size=batch, epochs=epoch)

            self.collect_data(env=self.env.env,
                              observation=obs, size=add_to_buffer)
            observation_batch, q_value_batch = self.batch_creation(
                batch_size=batch)

    def load_episode_buffer(self, episode_buffer: util.ReplayBuffer):
        observation = self.env.env.reset()
        while True:
            idx = episode_buffer.store_frame(observation)
            action = self.random_action(observation)
            observation, reward, done, _ = self.env.env.step(action)
            episode_buffer.store_effect(idx, action, reward, done)
            if reward == 20:
                break

    def train_model_episode(self, training_episode_count: int, test_episode_count: int):
        print('\n\n')
        print('Training DQN model (episode)')
        training_loss = []
        evaluation_metrics = []
        for i in tqdm(range(training_episode_count)):
            episode_buffer = util.ReplayBuffer(1e5, 1)
            self.load_episode_buffer(episode_buffer)
            state_batch, action_batch, reward_batch, new_state_batch, _ = episode_buffer.sample_all()

            # calculate Q Value of the model
            q_value_batch = np.array([
                self.q_value_calculation(state, next_state, reward, action)
                for state, action, reward, next_state in
                zip(state_batch, action_batch, reward_batch, new_state_batch)
            ])

            mean_training_loss, mse_training_loss = self.model.train_on_batch(
                state_batch, q_value_batch)
            training_loss.append([mean_training_loss, mse_training_loss])

            print(f'DQN Training; Step: {i}; MSE Training Loss: {mse_training_loss}; Mean Training Loss: {mean_training_loss}')

            if (i % 100) == 0:
                logging.info(
                    f'DQN Training; Step: {i}; MSE Training Loss: {mse_training_loss}; Mean Training Loss: {mean_training_loss}')
                evaluation_ep_reward, evaluation_ep_step = self.evaluate_model(
                    test_episode_count)
                logging.info(
                    f'Train Evaluation; Average Reward: {mean(evaluation_ep_reward)}; Average Step: {mean(evaluation_ep_step)}')
                evaluation_metrics.append([evaluation_ep_reward, evaluation_ep_step])
                self.save_model()
                self.epsilon = self.epsilon * 0.8
            else:
                logging.debug(
                    f'DQN Training; Step: {i}; MSE Training Loss: {mse_training_loss}; Mean Training Loss: {mean_training_loss}')
        return training_loss, evaluation_metrics

    def evaluate_model(self, episode_count: int):
        evaluation_ep_reward = []
        evaluation_ep_step = []
        print('\n\n')
        print('Evaluating DQN Model')
        for i in tqdm(range(episode_count)):
            episode_step_count = 0
            episode_reward = 0
            observation = self.evaluation_env.env.reset()
            while True:
                action = self.compute_action(observation=observation)
                observation, reward, done, _ = self.evaluation_env.env.step(
                    action)
                episode_reward += reward
                episode_step_count += 1
                if reward == 20:
                    break
            evaluation_ep_reward.append(episode_reward)
            evaluation_ep_step.append(episode_step_count)
            logging.info(
                f'Testing Iteration {i}; total Steps Taken {episode_step_count}; Rewards gained: {episode_reward}')
        return evaluation_ep_reward, evaluation_ep_step

    def save_model(self):
        self.model.save(self.model_save_path)

    def load_model(self):
        self.model.load_model(self.model_save_path)

if __name__ == "__main__":
    registerEnvironment()
    env = GymEnvironment(ENV_NAME)

    dqn_agent = DQN_Learning(
        env = env, 
        alpha = 0.9,
        epsilon = 0.5, 
        gamma = 0.2,
        learning_rate = 1e-3,
        hidden_layer_units = [100, 50], 
        dropout = [0.2, 0.2],
        initial_state_collection = 1e5, 
        model_save_path = 'ModelOutput/DQN_leanring_1'
    )

    dqn_agent.training_main(training_episode_count=500, test_episode_count=100)