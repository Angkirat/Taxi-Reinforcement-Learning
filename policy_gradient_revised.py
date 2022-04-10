import numpy as np
import torch
from gym_environment import GymEnvironment
from modified_taxi_environment import registerEnvironment, ENV_NAME

class Actor_Critic_model(torch.nn.Module):
    def __init__(self) -> None:
        super(Actor_Critic_model, self).__init__()
        num_states, hidden_units, num_actions = [500, 32, 6]
        self.hidden = torch.nn.Linear(num_states, hidden_units)
        self.action = torch.nn.Linear(hidden_units, num_actions)
        self.value = torch.nn.Linear(hidden_units, 1)
        self.saved_actions = []
        self.rewards = []
    
    def predict(self, state):
        intermediatory = torch.nn.functional.relu(self.hidden(state))
        policy = torch.nn.functional.softmax(self.action(intermediatory), dim=-1)
        qvalue = self.value(intermediatory)
        return policy, qvalue


class PollicyGradient_Learning:
    def __init__(
        self, env: GymEnvironment,
        num_states: int
    ) -> None:
        self.env = env
        self.ac_model = Actor_Critic_model()
        self.num_states = num_states
        self.episodic_rewards = []
        self.avg_episodic_rewards = []
        self.stdev_episodic_rewards = []
        self.best_avg_episodic_reward = -np.inf
        pass

    def compute_action(self, state):
        state = torch.from_numpy(state).float()
        probs, state_value = self.ac_model.predict(state)
        distribution = torch.distributions.Categorical(probs)
        action = distribution.sample()
        return action.item(), (distribution.log_prob(action), state_value)

    def encode_state(self,state: np.ndarray, batch_size:int):
        state_encoded = np.zeros((batch_size, self.num_states))
        state_encoded[np.arange(batch_size), state.astype(np.int64)] = 1
        return state_encoded

    def update_weights(self, saved_actions, reward_list):
        optimizer = torch.optim.Adam(self.ac_model.parameters(), lr=1e-3)
        gamma = 1   # Finite horizon
        eps = np.finfo(np.float32).eps.item()  # For stabilization
        R = 0
        # saved_actions = self.saved_actions
        policy_losses = []
        value_losses = []
        rewards = []
        for r in reward_list[::-1]:
            R = r + gamma * R
            rewards.insert(0, R)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        rewards = (rewards - rewards.mean()) / (rewards.std() + eps)    # Normalize - stabilize
        # Calculate losses
        for (log_prob, value), r in zip(saved_actions, rewards):
            reward = r - value.item()
            policy_losses.append(-log_prob * reward)
            value_losses.append(torch.nn.functional.smooth_l1_loss(value.squeeze(dim=1), torch.tensor([r])))
        # Apply optimization step
        optimizer.zero_grad()
        loss = torch.stack(policy_losses).sum() + torch.stack(value_losses).sum()
        loss.backward()
        optimizer.step()

    def start_training(self, training_episodes:int=20000):
        _, state_index = self.env.env.reset()
        episode_counter = 0
        frame_count = 0
        last_frame_count = 0
        episode_rewards = 0
        action_info_list = []
        reward_list = []
        while episode_counter < training_episodes:
            frame_count += 1
            state_encoded = self.encode_state(np.array([state_index]), 1)
            action, action_info = self.compute_action(state_encoded)
            (_, state_index), reward, done, _ = self.env.env.step(action)
            action_info_list.append(action_info)
            reward_list.append(reward)
            episode_rewards += reward
            if reward == 20:
                episode_counter += 1
                _, state_index = self.env.env.reset()
                self.episodic_rewards.append(episode_counter)
                episode_counter = 0

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
                    torch.save(self.ac_model, 'ModelOutput/torch_ActorCriticModel')
                
                if (episode_counter % 20) == 0:
                    print(
                        f'Episode {episode_counter}\tLast episode length: {(frame_count - last_frame_count)}\tAvg. Reward: {self.avg_episodic_rewards[-1]}')
                    print('Best avg. episodic reward:',
                          self.best_avg_episodic_reward)
                
                last_frame_count = frame_count

                self.update_weights(action_info_list, reward_list)
                action_info_list, reward_list = [], []
                

if __name__ == "__main__":
    print("\n\n\n")
    registerEnvironment()
    env = GymEnvironment(ENV_NAME)

    actor_critic_agent = PollicyGradient_Learning(env, 500)
    actor_critic_agent.start_training(20000)
