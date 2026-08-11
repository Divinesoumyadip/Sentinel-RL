"""
train_dqn.py

Double DQN + Prioritized Experience Replay training loop on
DetectorFaultControlEnv, with the pretrained LSTM plugged in as the
env's fault_prob_provider (this is the "LSTM encoder feeding RL state"
fine-tuning workflow called out in the JD).
"""
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn as nn
import random

from env.detector_env import DetectorFaultControlEnv
from models.dqn import QNetwork
from models.lstm_encoder import FaultLSTM
from train.replay_buffer import PrioritizedReplayBuffer

SENSOR_COLS = 6


class LSTMFaultProvider:
    """Wraps the pretrained LSTM so the env can call it as fault_prob_provider(window)."""
    def __init__(self, ckpt_path, device="cpu"):
        self.model = FaultLSTM()
        if os.path.exists(ckpt_path):
            self.model.load_state_dict(torch.load(ckpt_path, map_location=device))
            print(f"Loaded pretrained LSTM from {ckpt_path}")
        else:
            print("WARNING: no pretrained LSTM checkpoint found, using random-init weights.")
        self.model.eval()
        self.device = device

    def __call__(self, window):
        if len(window) < 3:
            return 0.0
        x = torch.tensor(np.array(window), dtype=torch.float32)
        _, fault_prob = self.model.encode(x)
        return fault_prob


def epsilon_by_frame(frame, eps_start=1.0, eps_end=0.05, eps_decay=15_000):
    return eps_end + (eps_start - eps_end) * np.exp(-1.0 * frame / eps_decay)


def train(
    n_episodes=400,
    max_steps=300,
    batch_size=64,
    gamma=0.99,
    lr=5e-4,
    target_update_every=500,
    buffer_capacity=50_000,
    device="cpu",
):
    ckpt = os.path.join(os.path.dirname(__file__), "..", "models", "lstm_pretrained.pt")
    fault_provider = LSTMFaultProvider(ckpt, device=device)
    env = DetectorFaultControlEnv(max_steps=max_steps, fault_prob_provider=fault_provider)

    obs_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n

    policy_net = QNetwork(obs_dim, n_actions).to(device)
    target_net = QNetwork(obs_dim, n_actions).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    opt = torch.optim.Adam(policy_net.parameters(), lr=lr)
    buffer = PrioritizedReplayBuffer(capacity=buffer_capacity)

    frame = 0
    episode_rewards = []

    for ep in range(n_episodes):
        obs, _ = env.reset(seed=None)
        ep_reward = 0.0

        for t in range(max_steps):
            eps = epsilon_by_frame(frame)
            if random.random() < eps:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    q = policy_net(torch.tensor(obs, dtype=torch.float32).unsqueeze(0))
                    action = int(q.argmax(dim=-1).item())

            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            buffer.push(obs, action, reward, next_obs, done)
            obs = next_obs
            ep_reward += reward
            frame += 1

            # ---- Double DQN learning step ----
            if len(buffer) >= batch_size:
                states, actions, rewards, next_states, dones, idxs, is_weights = \
                    buffer.sample(batch_size)

                states_t = torch.tensor(states)
                actions_t = torch.tensor(actions, dtype=torch.long)
                rewards_t = torch.tensor(rewards)
                next_states_t = torch.tensor(next_states)
                dones_t = torch.tensor(dones)
                weights_t = torch.tensor(is_weights)

                q_values = policy_net(states_t).gather(1, actions_t.unsqueeze(1)).squeeze(1)

                with torch.no_grad():
                    # Double DQN: action selection via policy net, evaluation via target net
                    next_actions = policy_net(next_states_t).argmax(dim=1)
                    next_q = target_net(next_states_t).gather(
                        1, next_actions.unsqueeze(1)).squeeze(1)
                    target_q = rewards_t + gamma * next_q * (1 - dones_t)

                td_errors = (q_values - target_q).detach().numpy()
                loss = (weights_t * nn.functional.smooth_l1_loss(
                    q_values, target_q, reduction="none")).mean()

                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(policy_net.parameters(), 10.0)
                opt.step()

                buffer.update_priorities(idxs, td_errors)

            if frame % target_update_every == 0:
                target_net.load_state_dict(policy_net.state_dict())

            if done:
                break

        episode_rewards.append(ep_reward)
        if (ep + 1) % 20 == 0:
            avg = np.mean(episode_rewards[-20:])
            print(f"episode {ep+1}/{n_episodes}  avg_reward(last20)={avg:.2f}  eps={eps:.3f}")

    ckpt_out = os.path.join(os.path.dirname(__file__), "..", "models", "dqn_policy.pt")
    torch.save(policy_net.state_dict(), ckpt_out)
    print(f"Saved trained DQN policy -> {ckpt_out}")
    return policy_net, episode_rewards


if __name__ == "__main__":
    train(n_episodes=int(os.environ.get("N_EPISODES", 400)))
