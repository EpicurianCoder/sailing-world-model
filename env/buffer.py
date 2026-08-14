# ==========================================
# CODE CITATION FOR: PPOBuffer
#
# I needed to adapt code and understand how the PPO buffere worked, I used the repos
# from the PPOAgent citation, as well as an OpenAI resource that was particularly useful
# for the finish_path() function
#
# ADAPTED FROM: Intro to PPO (https://spinningup.openai.com/en/latest/spinningup/rl_intro3.html)
# AUTHOR: OpenAI
# ==========================================
import numpy as np
import scipy.signal


class PPOBuffer:
    def __init__(self, gamma=0.99, lam=0.95):
        self.gamma = gamma
        self.lam = lam
        self.reset()

    def reset(self):
        self.obs_grids = []
        self.telemetry_vecs = []
        self.prev_actions = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []
        self.hidden_h_states = []
        self.hidden_c_states = []

    def store(self, grid, telemetry, prev_action, action, reward, value, log_prob, done, h, c):
        self.obs_grids.append(grid)
        self.telemetry_vecs.append(telemetry)
        self.prev_actions.append(prev_action)
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(value)
        self.log_probs.append(log_prob)
        self.dones.append(done)
        self.hidden_h_states.append(h)
        self.hidden_c_states.append(c)

    def finish_path(self, last_value=0.0):
        rewards = np.array(self.rewards + [last_value])
        values = np.array(self.values + [last_value])

        deltas = rewards[:-1] + self.gamma * values[1:] * (1.0 - np.array(self.dones)) - values[:-1]

        # Discount the advantages over time
        advantages = scipy.signal.lfilter([1], [1, -self.gamma * self.lam], deltas[::-1], axis=0)[::-1]

        returns = advantages + values[:-1]

        return np.array(advantages, dtype=np.float32), np.array(returns, dtype=np.float32)
