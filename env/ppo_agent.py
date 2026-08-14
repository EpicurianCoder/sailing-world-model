# ==========================================
# CODE CITATION FOR: PPOAgent (Proximal Policy Optimization)
# I needed to adapt code from the below repos to create my own version of the ppo_agent
#
# ADAPTED FROM: GitHub Repo (https://github.com/denismegerle/rl-ppo-agent/blob/master/ppo_agent.py)
# AUTHOR: Denis Megerle, 2020
# LICENSE: MIT
#
# ADAPTED FROM: GitHub Repo (https://github.com/tensorflow/agents/blob/master/tf_agents/agents/ppo/ppo_agent.py)
# AUTHOR: TensorFlow
# LICENSE: Apache-2.0
# ==========================================
import numpy as np
import tensorflow as tf
from thin_models import build_vision_model, build_memory_model, build_ppo_controller_and_critic


class PPOAgent:
    def __init__(self, clip_ratio=0.2, gamma=0.99, lam=0.95):
        self.clip_ratio = clip_ratio
        self.gamma = gamma
        self.lam = lam

        self.vision_model = build_vision_model()
        self.memory_lstm = build_memory_model()
        self.actor_critic = build_ppo_controller_and_critic()

        # Optimizers
        self.actor_optimizer = tf.keras.optimizers.Adam(learning_rate=3e-4)
        self.critic_optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)

        # Log Standard Deviation
        self.log_std = tf.Variable(tf.zeros((3,)), trainable=True, name="action_log_std")

    def get_action_and_value(self, obs_grid, telemetry_vec, prev_actions, hidden_h, hidden_c):
        # Vision
        grid_tensor = tf.convert_to_tensor(np.expand_dims(obs_grid, axis=0), dtype=tf.float32)
        latent_vec = self.vision_model(grid_tensor)

        # Memory
        telem_tensor = tf.convert_to_tensor(np.expand_dims(telemetry_vec, axis=0), dtype=tf.float32)
        act_tensor = tf.convert_to_tensor(np.expand_dims(prev_actions, axis=0), dtype=tf.float32)

        lstm_out, new_h, new_c = self.memory_lstm([latent_vec, telem_tensor, act_tensor, hidden_h, hidden_c])

        # Controller
        action_mean, value = self.actor_critic([latent_vec, lstm_out])

        std = tf.exp(self.log_std)
        action_distribution = tf.random.normal(shape=action_mean.shape, mean=action_mean, stddev=std)

        variance = tf.square(std)
        log_prob = -0.5 * tf.square(action_distribution - action_mean) / variance - 0.5 * tf.math.log(2 * np.pi * variance)
        log_prob_sum = tf.reduce_sum(log_prob, axis=-1)

        sampled_action = tf.clip_by_value(action_distribution, -1.0, 1.0)

        return np.squeeze(sampled_action), np.squeeze(value), np.squeeze(log_prob_sum), new_h, new_c

    # Calculate GAE (Generalized Advantage Estimation)
    def compute_advantages(self, rewards, values, last_value, dones):
   
        advantages = np.zeros_like(rewards, dtype=np.float32)
        last_gae_lam = 0

        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_non_terminal = 1.0 - dones[-1]
                next_values = last_value
            else:
                next_non_terminal = 1.0 - dones[t]
                next_values = values[t+1]

            delta = rewards[t] + self.gamma * next_values * next_non_terminal - values[t]
            advantages[t] = last_gae_lam = delta + self.gamma * self.lam * next_non_terminal * last_gae_lam

        returns = advantages + values
        return advantages, returns

    def save_models(self, path_prefix="weights/ppo"):
        import os
        os.makedirs(os.path.dirname(path_prefix), exist_ok=True)

        self.vision_model.save_weights(f"{path_prefix}_vision.weights.h5")
        self.memory_lstm.save_weights(f"{path_prefix}_memory.weights.h5")
        self.actor_critic.save_weights(f"{path_prefix}_actor_critic.weights.h5")

        # Save the exploration standard deviation variable
        np.save(f"{path_prefix}_log_std.npy", self.log_std.numpy())
        print(f"--- Models safely saved to {path_prefix} ---")

    def load_models(self, path_prefix="weights/ppo"):
        try:
            self.vision_model.load_weights(f"{path_prefix}_vision.weights.h5")
            self.memory_lstm.load_weights(f"{path_prefix}_memory.weights.h5")
            self.actor_critic.load_weights(f"{path_prefix}_actor_critic.weights.h5")

            loaded_std = np.load(f"{path_prefix}_log_std.npy")
            self.log_std.assign(loaded_std)
            print(f"--- Models successfully loaded from {path_prefix} ---")
        except Exception as e:
            print(f"--- Could not load models: {e} ---")

    # Calculate log probability of an action under a Normal distribution
    def _get_log_prob(self, action, mean, std):
        """."""
        variance = tf.square(std)
        log_prob = -0.5 * tf.square(action - mean) / variance - 0.5 * tf.math.log(2 * np.pi * variance)
        return tf.reduce_sum(log_prob, axis=-1)

    # Reshape the flat episode data into sequences and processes in batches.
    # length = `seq_len`
    def update_networks(self, buffer, advantages, returns, seq_len=16, epochs=4):
        actor_losses_list = []
        critic_losses_list = []
        # Normalize advantages for stability
        advantages = (advantages - np.mean(advantages)) / (np.std(advantages) + 1e-8)

        num_steps = len(buffer.actions)
        # Drop remainder steps that don't fit perfectly into seq_len
        num_chunks = num_steps // seq_len 

        if num_chunks == 0:
            return None, None

        valid_steps = num_chunks * seq_len

        # Reshape (Batch, Seq_Len, Features)
        b_grids = np.reshape(buffer.obs_grids[:valid_steps], (num_chunks, seq_len, 3, 256, 256))
        b_telem = np.reshape(buffer.telemetry_vecs[:valid_steps], (num_chunks, seq_len, -1))
        b_prev_a = np.reshape(buffer.prev_actions[:valid_steps], (num_chunks, seq_len, -1))
        b_actions = np.reshape(buffer.actions[:valid_steps], (num_chunks, seq_len, -1))
        b_old_log_probs = np.reshape(buffer.log_probs[:valid_steps], (num_chunks, seq_len))
        b_adv = np.reshape(advantages[:valid_steps], (num_chunks, seq_len))
        b_ret = np.reshape(returns[:valid_steps], (num_chunks, seq_len))

        b_h = np.array(buffer.hidden_h_states)[0:valid_steps:seq_len]
        b_c = np.array(buffer.hidden_c_states)[0:valid_steps:seq_len]

        b_h = np.squeeze(b_h, axis=1)
        b_c = np.squeeze(b_c, axis=1)

        t_grids = tf.convert_to_tensor(b_grids, dtype=tf.float32)
        t_telem = tf.convert_to_tensor(b_telem, dtype=tf.float32)
        t_prev_a = tf.convert_to_tensor(b_prev_a, dtype=tf.float32)
        t_actions = tf.convert_to_tensor(b_actions, dtype=tf.float32)
        t_old_log_probs = tf.convert_to_tensor(b_old_log_probs, dtype=tf.float32)
        t_adv = tf.convert_to_tensor(b_adv, dtype=tf.float32)
        t_ret = tf.convert_to_tensor(b_ret, dtype=tf.float32)

        t_h = tf.convert_to_tensor(b_h, dtype=tf.float32)
        t_c = tf.convert_to_tensor(b_c, dtype=tf.float32)

        # Optimization
        for epoch in range(epochs):
            with tf.GradientTape(persistent=True) as tape:
                # Vision
                flat_grids = tf.reshape(t_grids, (-1, 3, 256, 256))
                flat_latents = self.vision_model(flat_grids, training=True)
                b_latents = tf.reshape(flat_latents, (num_chunks, seq_len, 128))

                # Memory
                curr_h = t_h
                curr_c = t_c
                lstm_outputs = []

                # Step through time
                for t in range(seq_len):
                    lat_t = b_latents[:, t, :]
                    tel_t = t_telem[:, t, :]
                    act_t = t_prev_a[:, t, :]

                    out_t, curr_h, curr_c = self.memory_lstm(
                        [lat_t, tel_t, act_t, curr_h, curr_c], training=True
                    )
                    lstm_outputs.append(out_t)

                lstm_seq_out = tf.stack(lstm_outputs, axis=1)

                flat_latents_ac = tf.reshape(b_latents, (-1, 128))
                flat_lstm_out = tf.reshape(lstm_seq_out, (-1, 128))

                # Pass the flat 2D tensors
                flat_action_means, flat_values = self.actor_critic([flat_latents_ac, flat_lstm_out], training=True)
                action_means = tf.reshape(flat_action_means, (num_chunks, seq_len, 3))
                values = tf.reshape(flat_values, (num_chunks, seq_len))

                # PPO Math
                std = tf.exp(self.log_std)
                new_log_probs = self._get_log_prob(t_actions, action_means, std)

                ratio = tf.exp(new_log_probs - t_old_log_probs)
                clip_adv = tf.clip_by_value(ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio) * t_adv

                actor_loss = -tf.reduce_mean(tf.minimum(ratio * t_adv, clip_adv))
                critic_loss = tf.reduce_mean(tf.square(values - t_ret))

                entropy = tf.reduce_mean(0.5 + 0.5 * tf.math.log(2.0 * np.pi * tf.square(std)))
                entropy_bonus = -0.01 * entropy

                total_actor_loss = actor_loss + entropy_bonus

                actor_losses_list.append(total_actor_loss.numpy())
                critic_losses_list.append(critic_loss.numpy())

            # Apply Gradients
            actor_trainable = (
                self.vision_model.trainable_variables +
                self.memory_lstm.trainable_variables +
                self.actor_critic.trainable_variables + [self.log_std]
            )

            actor_grads = tape.gradient(total_actor_loss, actor_trainable)
            critic_grads = tape.gradient(critic_loss, self.actor_critic.trainable_variables)

            # Filter out None gradients
            # Actor loss mustn't affect Critic dense layers
            # Critic loss mustn't affect Actor dense layers
            actor_grads_and_vars = [(g, v) for g, v in zip(actor_grads, actor_trainable) if g is not None]
            critic_grads_and_vars = [(g, v) for g, v in zip(critic_grads, self.actor_critic.trainable_variables) if g is not None]

            self.actor_optimizer.apply_gradients(actor_grads_and_vars)
            self.critic_optimizer.apply_gradients(critic_grads_and_vars)

            del tape

        return np.mean(actor_losses_list), np.mean(critic_losses_list)

    def get_deterministic_action(self, obs_grid, telemetry_vec, prev_actions, hidden_h, hidden_c):
        grid_tensor = tf.convert_to_tensor(np.expand_dims(obs_grid, axis=0), dtype=tf.float32)
        latent_vec = self.vision_model(grid_tensor, training=False)

        telem_tensor = tf.convert_to_tensor(np.expand_dims(telemetry_vec, axis=0), dtype=tf.float32)
        act_tensor = tf.convert_to_tensor(np.expand_dims(prev_actions, axis=0), dtype=tf.float32)

        # Pass through LSTM
        lstm_out, new_h, new_c = self.memory_lstm(
            [latent_vec, telem_tensor, act_tensor, hidden_h, hidden_c], training=False
        )

        action_mean, _ = self.actor_critic([latent_vec, lstm_out], training=False)
        deterministic_action = tf.clip_by_value(action_mean, -1.0, 1.0)

        return np.squeeze(deterministic_action), new_h, new_c
