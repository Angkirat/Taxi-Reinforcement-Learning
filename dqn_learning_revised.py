import random
from turtle import shape
from typing import Any
import tensorflow as tf
import numpy as np

import util
from gym_environment import GymEnvironment
from modified_taxi_environment import registerEnvironment, ENV_NAME


class DQN_Learning:
    def __init__(
        self, env: GymEnvironment, discount: float,
        isOneHot: bool, state_dim: int, hidden_layer_units: list, add_dropout: bool,
        dropout_units: list, optimizer: Any, random_act_iterations: int, min_random_act_ep: float
    ) -> None:
        self.env = env
        self.isOneHot = isOneHot
        self.state_dim = state_dim
        self.hidden_layer_units = hidden_layer_units
        self.add_droput = add_dropout
        self.dropout_units = dropout_units
        self.optimizer = optimizer
        self.model = self.create_model()
        self.target_model = self.create_model()
        self.random_act_iterations = random_act_iterations
        self.min_random_act_ep = min_random_act_ep
        self.model_store_file = 'DQN_Learning-' + \
            ('onehot' if isOneHot else 'state')
        self.episodic_rewards = []
        self.avg_episodic_rewards = []
        self.stdev_episodic_rewards = []
        self.acc_episodic_reward = 0.0
        self.best_avg_episodic_reward = -np.inf
        self.replay_buffer = util.ReplayBuffer(200_000, 1)
        self.discount = discount
        pass

    def create_model(self):
        model = tf.keras.Sequential(name="DQNModel")
        if self.isOneHot:
            print("Creating a OneHot model input must be of shape", self.state_dim)
            model.add(tf.keras.Input(shape=self.state_dim))
        else:
            print("Creating a state Model input shape must be", self.env.obs_shape)
            model.add(tf.keras.Input(shape=self.env.obs_shape))
        for i, (units, drop) in enumerate(zip(self.hidden_layer_units, self.dropout_units)):
            model.add(tf.keras.layers.Dense(
                units,
                activation=tf.keras.activations.relu,
                name=f"Hidden_Layer_{(i+1)}"
            ))
            if self.add_droput:
                model.add(tf.keras.layers.Dropout(
                    drop,
                    name=f"Dropout_Layer_{(i+1)}"
                ))
        model.add(tf.keras.layers.Dense(
            self.env.action_size,
            activation=None,
            name="outputLayer"
        ))
        model.compile(optimizer=self.optimizer, loss="mse", metrics=["mae"])
        return model

    def save_model(self):
        self.model.save(self.model_store_file)

    def load_model(self):
        self.model.load_model(self.model_store_file)

    def onehotEncode_state(self, states: np.ndarray, batch_size:int) -> np.ndarray:
        states_encoded = np.zeros((batch_size, self.state_dim))
        states_encoded[np.arange(batch_size), states.astype(np.int64)] = 1
        return states_encoded

    def compute_action(self, state: np.ndarray) -> int:
        return int(self.model.predict(state)[0].argmax())

    def random_action(self, iteration, state: np.ndarray):
        fraction = min(1.0, float(iteration)/self.random_act_iterations)
        epsilon = 1 + fraction*(self.min_random_act_ep - 1)
        sample = random.random()  # nosec
        if sample <= epsilon:
            return self.env.sample_action(), epsilon
        else:
            return self.compute_action(state), epsilon

    def start_training(self, training_episodes: int, start_training: int, batch_size: int, target_update_freq: int):
        state_detail, state_idx = self.env.env.reset()
        num_param_updates = 0
        last_frame_count = 0
        episode_counter = 0
        frame_count = 0
        while episode_counter < training_episodes:
            if self.replay_buffer.num_in_buffer < start_training:
                action = self.env.sample_action()
                eps = 1
            else:
                if self.isOneHot:
                    encoded_state = self.onehotEncode_state(np.array([state_idx]), 1)
                else:
                    encoded_state = np.array([state_detail])
                action, eps = self.random_action(frame_count, encoded_state)

            buffer_idx = self.replay_buffer.store_frame(
                (np.array(state_idx) if self.isOneHot else state_detail))
            (state_detail, state_idx), reward, done, _ = self.env.env.step(action)
            self.acc_episodic_reward += reward

            self.replay_buffer.store_effect(buffer_idx, action, reward, done)
            frame_count += 1
            if done:
                state_detail, state_idx = self.env.env.reset()

                episode_counter += 1
                self.episodic_rewards.append(self.acc_episodic_reward)
                self.acc_episodic_reward = 0

                if len(self.episodic_rewards) <= 10:
                    self.avg_episodic_rewards.append(
                        np.mean(self.episodic_rewards))
                    if len(self.episodic_rewards) > 2:
                        self.stdev_episodic_rewards.append(
                            np.std(self.episodic_rewards))
                else:
                    self.avg_episodic_rewards.append(
                        np.mean(self.episodic_rewards[-10:]))
                    self.avg_episodic_rewards.append(
                        np.mean(self.episodic_rewards[-10:]))

                if self.avg_episodic_rewards[-1] > self.best_avg_episodic_reward:
                    self.best_avg_episodic_reward = self.avg_episodic_rewards[-1]
                    self.save_model()

                if (episode_counter % 20) == 0:
                    print(
                        f'Episode {episode_counter}\tLast episode length: {(frame_count - last_frame_count)}\tAvg. Reward: {self.avg_episodic_rewards[-1]}\tEpsilon: {np.round(eps,4)}\t')
                    print('Best avg. episodic reward:',
                          self.best_avg_episodic_reward)

                last_frame_count = frame_count
                pass

            if frame_count > start_training and self.replay_buffer.can_sample(batch_size=batch_size):
                state_batch, act_batch, reward_batch, new_state_batch, done_mask_batch = self.replay_buffer.sample(
                    batch_size)

                # current State q value
                # current_qvalue = self.model.predict(state_batch)[range(batch_size), act_batch]
                state_batch = self.onehotEncode_state(state_batch, batch_size)
                new_state_batch = self.onehotEncode_state(new_state_batch, batch_size)

                next_state_qvalue = self.target_model.predict(
                    new_state_batch).max(1)

                reversed_done_mask = 1 - done_mask_batch
                next_state_qvalue = reversed_done_mask * next_state_qvalue

                qvalue_computed = reward_batch + (self.discount * next_state_qvalue)

                qvalue_computed = tf.convert_to_tensor(np.array(qvalue_computed))

                with tf.GradientTape() as tape:
                    y_pred = self.model(state_batch, training=True)  # Forward pass
                    y_pred = tf.gather_nd(y_pred, list(enumerate(act_batch)))
                    # Compute our own loss
                    
                    loss = self.model.compiled_loss(
                        qvalue_computed, y_pred, regularization_losses=self.model.losses)

                gradients = tape.gradient(loss, self.model.trainable_variables)
                # Update weights
                self.model.optimizer.apply_gradients(
                    zip(gradients, self.model.trainable_variables))
                # Update metrics (includes the metric that tracks the loss)
                self.model.compiled_metrics.update_state(
                    qvalue_computed, y_pred)

                # print({m.name: m.result() for m in self.metrics})

                num_param_updates += 1

                if (num_param_updates % target_update_freq) == 0:
                    self.target_model.set_weights(self.model.get_weights())
                    pass


if __name__ == "__main__":
    registerEnvironment()
    env = GymEnvironment(ENV_NAME)

    optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.00025, rho=0.95)

    dqn_agent = DQN_Learning(
        env, discount=0.2, isOneHot=True, state_dim=500, hidden_layer_units=[100, 50], add_dropout=True,
        dropout_units=[0.2, 0.2], optimizer=optimizer, random_act_iterations=350_000, min_random_act_ep=0.1
    )

    print("\n\n\n")
    print(dqn_agent.model.summary())
    print("\n\n\n")

    dqn_agent.start_training(
        training_episodes=3_000, start_training=50_000, batch_size=32, target_update_freq=10_000)
