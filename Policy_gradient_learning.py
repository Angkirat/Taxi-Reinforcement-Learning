import pickle  # nosec
import logging
import numpy as np
from tqdm import tqdm
import tensorflow as tf
from statistics import mean
from datetime import datetime
import tensorflow_probability as tfp

import util
from gym_environment import GymEnvironment
from modified_taxi_environment import registerEnvironment, ENV_NAME

logdir = "logs/Policy/" + datetime.now().strftime("%Y%m%d-%H%M%S")
file_writer = tf.summary.create_file_writer(logdir + "/metrics")
file_writer.set_as_default()
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)

logging.basicConfig(
    format='%(levelname)s:%(asctime)s:%(message)s', datefmt='%m/%d/%Y %I:%M:%S %p',
    filename='logs/Policy-Learning.log', encoding='utf-8', level=(logging.INFO)
)


class PolicyGradient_learning:
    def __init__(self,
                 env: GymEnvironment,
                 model_save_path: str = 'ModelOutput/Policy_leanring',
                 learning_rate: float = 1e-3,
                 hidden_layer_units: list = [100, 50],
                 dropout: list = [0.2, 0.2],
                 gamma: float = 0.2,
                 epsilon: float = 0.5
                 ) -> None:
        self.training_env = env
        self.evaluation_env = env
        self.model = None
        self.optimizer = tf.optimizers.Adam(learning_rate=learning_rate)
        self.epsilon = epsilon
        self.discount = gamma
        self.model_save_path = model_save_path
        self.hidden_layer_units = hidden_layer_units
        self.dropout = dropout

    def training_main(self, training_episode_size=50000):
        self.model = self.create_model()
        training_loss, evaluation_list = self.train(training_episode_size)
        evaluation_ep_reward, evaluation_ep_step = self.evaluate_model(200)
        print(
            f'Final Evaluation; Reward: {mean(evaluation_ep_reward)}; step: {mean(evaluation_ep_step)}')
        to_save = {
            'training_loss': training_loss,
            'evaluation_list': evaluation_list,
            'evaluation_ep_reward': evaluation_ep_reward,
            'evaluation_ep_step': evaluation_ep_step
        }
        pickle.dump(to_save, open('ModelOutput/Policy-Metrics_1.pkl', 'wb'))
        pass

    def create_model(self):
        model = tf.keras.Sequential(name="Policy_Gradient")
        model.add(tf.keras.Input(shape=self.training_env.obs_shape))
        for i, (units, drop) in enumerate(zip(self.hidden_layer_units, self.dropout)):
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

    def train(self, training_episode: int):
        try:
            training_iteration_count = 0
            # training_loss = []
            evaluation_list = []
            for i in tqdm(range(training_episode)):
                buffer = util.ReplayBuffer(1e5, 1)
                self.collect_data(buffer)
                state_batch, reward_batch, action_batch = self.get_batch(buffer)
                episode_loss = []
                state_count = state_batch.shape[0]
                # print(f'Iteration {i} State size: ', state_count)
                state_itr = 0
                for state, reward, action in zip(state_batch, reward_batch, action_batch):
                    # print(f'State Iterator: {state_itr} of {state_count}')
                    state_itr += 1
                    with tf.GradientTape() as tape:
                        pred = self.model(np.array([state]), training=True)
                        loss = self.model_loss(pred, action, reward)
                        episode_loss.append(loss)
                    grads = tape.gradient(loss, self.model.trainable_variables)
                    self.optimizer.apply_gradients(
                        zip(grads, self.model.trainable_variables))
                # training_loss.append(mean(episode_loss))
                self.epsilon = self.epsilon - (self.epsilon * 0.2)
                if (i % 100) == 0:
                    logging.info(
                        f'Policy Training; Training Iteration: {training_iteration_count}; Training Loss: {loss}')
                    evaluation_ep_reward, evaluation_ep_step = self.evaluate_model(2)
                    evaluation_list.append(
                        [evaluation_ep_reward, evaluation_ep_step])
                    self.save_model()
        except KeyboardInterrupt:
            print('Ended the file execution')
        except Exception as e:
            print(e)
        finally:
            self.save_model()
        return evaluation_list

    def compute_action(self, observation: np.ndarray):
        prob = self.model.predict(np.array([observation]))
        dist = tfp.distributions.Categorical(prob, dtype=tf.float32)
        action = dist.sample()
        return int(action.numpy()[0])

    def random_action(self, observation: np.ndarray):
        if util.flipCoin(self.epsilon):
            return self.training_env.sample_action()
        return self.compute_action(observation)

    def get_batch(self, buffer: util.ReplayBuffer):
        state_batch, action_batch, reward_batch, _, _ = buffer.sample_all()
        sum_reward = 0
        reward_list = []
        reward_batch = np.flip(reward_batch)
        for r in reward_batch:
            sum_reward = r + (self.discount * sum_reward)
            reward_list.append(sum_reward)
        reward_list.reverse()
        return state_batch, reward_list, action_batch

    def evaluate_model(self, episode_count: int):
        evaluation_ep_reward = []
        evaluation_ep_step = []
        for i in tqdm(range(episode_count)):
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
            logging.info(
                f'Policy Testing; Episode # {i}; Episodde Reward: {episode_reward}; Episode Step: {episode_step_count}')
        return evaluation_ep_reward, evaluation_ep_step

    def collect_data(self, buffer: util.ReplayBuffer):
        episode_reward = 0
        episode_step = 0
        observation = self.training_env.env.reset()
        while True:
            indx = buffer.store_frame(observation)
            action = self.random_action(observation)
            observation, reward, done, _ = self.training_env.env.step(action)
            buffer.store_effect(indx, action, reward, done)
            episode_reward += reward
            episode_step += 1
            if done:
                break
            pass
        # print(f'Training batch: Reward: {episode_reward} in {episode_step} steps')
        pass

    def save_model(self):
        self.model.save(self.model_save_path)

    def load_model(self):
        self.model.load_model(self.model_save_path)


if __name__ == "__main__":

    registerEnvironment()
    env = GymEnvironment(ENV_NAME)

    policy_agent = PolicyGradient_learning(
        env=env,
        model_save_path='ModelOutput/Policy_leanring_1',
        learning_rate=1e-3,
        hidden_layer_units=[300, 100],
        dropout=[0.2, 0.2],
        gamma=0.2,
        epsilon=1
    )

    policy_agent.training_main(training_episode_size=1000)

    policy_agent.save_model()

    pass
