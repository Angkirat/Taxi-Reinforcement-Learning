import gym
import numpy as np
from tqdm import tqdm
import tensorflow as tf
import tensorflow_probability as tfp

from gym_environment import GymEnvironment
import util


class PolicyGradient_learning:
    def __init__(self,
                 env: GymEnvironment,
                 learning_rate: float = 1e-5,
                 hidden_layer_units: list = [100, 50], 
                 dropout: list = [0.5, 0.5]
                 ) -> None:
        self.training_env = env
        self.evaluation_env = env
        self.model = self.create_model(hidden_layer_units, dropout)
        self.buffer = util.ReplayBuffer(1e5, 1)
        self.optimizer = tf.optimizers.Adam(learning_rate=learning_rate)

    def create_model(self, hidden_layer_units: list, dropout: list):
        model = tf.keras.Sequential(name="Policy_Gradient")
        model.add(tf.keras.Input(shape=self.training_env.observation_shape))
        for i, (units, drop) in enumerate(zip(hidden_layer_units, dropout)):
            model.add(tf.keras.layers.Dense(
                units,
                activation=tf.keras.activations.relu,
                name=f"Hidden_Layer_{(i+1)}"
            ))
            model.add(tf.keras.layers.Dropout(
                drop, name=f"Dropout_Layer_{(i+1)}"
            ))
        model.add(tf.keras.layers.Dense(
            self.training_env.action_size,
            activation=tf.keras.activations.softmax,
            name="Output_Layer"
        ))
        return model

    def model_loss(self, prob, action, reward):
        dist = tfp.distributions.Categorical(prob, dtype=tf.float32)
        log_prob = dist.log_prob(action)
        return (-1 * log_prob * reward)

    def train(self, state_batch, reward_batch, action_batch):
        sum_reward = 0
        reward_list = []
        reward_batch.reverse()
        for r in reward_batch:
            sum_reward = r + (self.discount * sum_reward)
            reward_list.append(sum_reward)
        reward_list.reverse()
        for state, reward, action in tqdm(zip(state_batch, reward_list, action_batch)):
            with tf.GradientTape() as tape:
                pred = self.model(np.array([state]), training=True)
                loss = self.model_loss(pred, action, reward)
            grads = tape.gradient(loss, self.model.trainable_variables)
            self.optimizer.apply_gradients(
                zip(grads, self.model.trainable_variables))

    def compute_action(self, observation: np.ndarray):
        prob = self.model.predict(np.array(observation))
        dist = tfp.distributions.Categorical(prob, dtype=tf.float32)
        action = dist.sample()
        return int(action.numpy()[0])

    def random_action(self, observation: np.ndarray):
        if util.flipCoin(self.epsilon):
            return self.env.sample_action()
        return self.compute_action(observation)

    def get_batch(self):
        state_batch, action_batch, reward_batch, _, _ = self.buffer.sample_all()
        sum_reward = 0
        reward_list = []
        reward_batch.reverse()
        for r in reward_batch:
            sum_reward = r + (self.discount * sum_reward)
            reward_list.append(sum_reward)
        reward_list.reverse()
        return state_batch, reward_list, action_batch

    def evaluate_model(self, episode_count: int):
        evaluation_ep_reward = []
        evaluation_ep_step = []
        for _ in range(episode_count):
            observation = self.evaluation_env.env.reset()
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
        return evaluation_ep_reward, evaluation_ep_step

    def collect_data(self, size: int):
        for _ in tqdm(range(size)):
            observation = self.training_env.env.reset()
            while True:
                indx = self.buffer.store_frame(observation)
                action = self.random_action(observation)
                observation, reward, done, _ = self.training_env.step(action)
                self.buffer.store_effect(indx, action, reward, done)
                if done:
                    break
                pass
            pass
