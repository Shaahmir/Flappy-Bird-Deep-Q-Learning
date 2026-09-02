import os
import yaml
import random
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
import flappy_bird_gymnasium

from collections import deque
from dqn import DQN
from experience_replay import ReplayMemory

device = "cuda" if torch.cuda.is_available() else "cpu"
RUN_DIR = "runs"
os.makedirs(RUN_DIR, exist_ok = True)

class Agent:

    def __init__(self, param_set: str):

        self.param_set = param_set
        
        with open("config.yaml", "r") as f:
            params = yaml.safe_load(f)[param_set]
        
        self.alpha = params["alpha"]
        self.gamma = params["gamma"]
        self.epsilon_init = params["epsilon_init"]
        self.epsilon_min = params["epsilon_min"]
        self.epsilon_decay = params["epsilon_decay"]
        self.replay_memory_size = params["replay_memory_size"]
        self.batch_size = params["batch_size"]
        self.reward_threshold = params.get("reward_threshold", 10000)
        self.soft_update_tau = params.get("soft_update_tau", 0.005)
        self.hidden_dim = params.get("hidden_dim", 256)
        self.dueling = params.get("dueling", True)
        self.double_dqn = params.get("double_dqn", True)
        self.episodes = params.get("episodes", 5000)
        self.save_every = params.get("save_every", 100)

        self.loss_fn = nn.SmoothL1Loss()
        self.optimizer = None

        self.LOG_FILE = os.path.join(RUN_DIR, f"{self.param_set}.log")
        self.MODEL_FILE = os.path.join(RUN_DIR, f"{self.param_set}.pt")
        self.BEST_MODEL_FILE = os.path.join(RUN_DIR, f"{self.param_set}_best.pt")

    def soft_update(self, target: nn.Module, source: nn.Module, tau: float):

        for target_param, source_param in zip(target.parameters(), source.parameters()):

            target_param.data.copy_(
                tau * source_param.data + (1.0 - tau) * target_param.data
            )

    def optimize(self, mini_batch, policy_dqn, target_dqn):
        
        states, actions, next_states, rewards, terminations = zip(*mini_batch)


        states      = torch.stack(states).to(device)
        actions     = torch.stack(actions).to(device)
        next_states = torch.stack(next_states).to(device)
        rewards     = torch.stack(rewards).to(device)
        terminations = torch.tensor(terminations, dtype=torch.float32, device=device)


        with torch.no_grad():

            if self.double_dqn:
                next_actions = policy_dqn(next_states).argmax(dim=1, keepdim=True)
                next_q = target_dqn(next_states).gather(1, next_actions).squeeze(1)

            else:
                next_q = target_dqn(next_states).max(dim=1)[0]

            target_q = rewards + (1.0 - terminations) * self.gamma * next_q


        current_q = policy_dqn(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        loss = self.loss_fn(current_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy_dqn.parameters(), max_norm = 10.0)
        self.optimizer.step()

        return loss.item()

    def run(self, is_training: bool = True, render: bool = False):
        
        env = gym.make(
            "FlappyBird-v0",
            render_mode = "human" if render else None,
            use_lidar = False,
        )

        num_states = env.observation_space.shape[0]
        num_actions = env.action_space.n

        policy_dqn = DQN(num_states, num_actions, self.hidden_dim, self.dueling).to(device)
        target_dqn = DQN(num_states, num_actions, self.hidden_dim, self.dueling).to(device)
        target_dqn.load_state_dict(policy_dqn.state_dict())
        target_dqn.eval()

        if is_training:

            memory = ReplayMemory(self.replay_memory_size)
            epsilon = self.epsilon_init
            self.optimizer = optim.Adam(policy_dqn.parameters(), lr=self.alpha)
            best_reward = float("-inf")
            reward_history = deque(maxlen=100)
            total_steps = 0

        else:

            if not os.path.exists(self.BEST_MODEL_FILE):
                raise FileNotFoundError(f"No model found at {self.BEST_MODEL_FILE}")

            policy_dqn.load_state_dict(torch.load(self.BEST_MODEL_FILE, map_location=device))
            policy_dqn.eval()
            epsilon = 0.0

        for episode in range(1, self.episodes + 1 if is_training else 100):

            state, _ = env.reset()
            state = torch.tensor(state, dtype = torch.float32, device = device)
            episode_reward = 0.0
            terminated = False
            truncated = False
            losses = []

            while not (terminated or truncated) and episode_reward < self.reward_threshold:
 
                if is_training and random.random() < epsilon:
                    action = env.action_space.sample()

                else:
                    with torch.no_grad():
                        q_values = policy_dqn(state.unsqueeze(0))
                        action = q_values.argmax().item()

                action_t = torch.tensor(action, dtype = torch.long, device = device)

                next_state, reward, terminated, truncated, info = env.step(action)
                episode_reward += reward

                next_state_t = torch.tensor(next_state, dtype = torch.float32, device = device)
                reward_t = torch.tensor(reward, dtype = torch.float32, device = device)

                if is_training:

                    memory.append((state, action_t, next_state_t, reward_t, terminated or truncated))
                    total_steps += 1

                    if len(memory) >= self.batch_size:
                        batch = memory.sample(self.batch_size)
                        loss = self.optimize(batch, policy_dqn, target_dqn)
                        losses.append(loss)

                        self.soft_update(target_dqn, policy_dqn, self.soft_update_tau)

                state = next_state_t

            avg_loss = np.mean(losses) if losses else 0.0

            if is_training:

                epsilon = max(epsilon * self.epsilon_decay, self.epsilon_min)
                reward_history.append(episode_reward)
                avg_reward = np.mean(reward_history)

                log_msg = f"Ep: {episode:5d} | Reward: {episode_reward:7.1f} | Avg Reward: {avg_reward:7.1f} | Epsilon: {epsilon:.3f} | Loss {avg_loss:.4f} | Steps {total_steps}"
                print(log_msg)

                with open(self.LOG_FILE, "a") as f:
                    f.write(log_msg + "\n")

                if episode_reward > best_reward:

                    best_reward = episode_reward
                    torch.save(policy_dqn.state_dict(), self.BEST_MODEL_FILE)
                    print(f"New best model saved! (Reward = {best_reward:.1f})")

                if episode % self.save_every == 0:
                    torch.save(policy_dqn.state_dict(), self.MODEL_FILE)

            else:
                print(f"Episode {episode} | Reward {episode_reward:.1f} | Score {info.get('score', 0)}")

        env.close()
