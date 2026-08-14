import time
import datetime
import numpy as np
import tensorflow as tf
from environment import SailingEnvironment
from physics import SailingPhysics
from ppo_agent import PPOAgent
from buffer import PPOBuffer


def scale_actions_for_env_old(raw_actions):
    return {
        'boat_angle': ((raw_actions[0] + 1.0) / 2.0) * 360.0,
        'sail_angle': raw_actions[1] * 90.0,
        'sail_size': ((raw_actions[2] + 1.0) / 2.0) * 3.0 + 3.0
    }


def scale_actions_for_env(raw_actions):
    return {
        'steer_axis': raw_actions[0],  # [-1.0, 1.0] Left or Right Turn
        'sail_trim_axis': raw_actions[1],  # [-1.0, 1.0] Increase or Decrease sail size
        'sail_size': ((raw_actions[2] + 1.0) / 2.0) * 3.0 + 3.0  # Size size is absolute value
    }


def run_test_training(duration_minutes=20):
    physics = SailingPhysics()
    env = SailingEnvironment(physics)
    agent = PPOAgent()
    buffer = PPOBuffer()

    print("\n*** MODEL ARCHITECTURE ***")
    agent.vision_model.summary()
    agent.memory_lstm.summary()
    agent.actor_critic.summary()
    print("---------------------------\n")

    # Attempt to load previous brains if they exist
    agent.load_models("saved_models/sailing_agent")

    # Setup TensorBoard logging
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = f"logs/ppo_sailing/{current_time}"
    summary_writer = tf.summary.create_file_writer(log_dir)

    print(f"--- Starting {duration_minutes}-Minute Training Run ---")

    start_time = time.time()
    max_duration_seconds = duration_minutes * 60
    ep = 0

    # Loop until the clock runs out
    while (time.time() - start_time) < max_duration_seconds:
        obs_grid, env_state = env.reset()
        telemetry_vec = env_state.to_telemetry_vector()

        prev_actions = np.zeros(3, dtype=np.float32)
        hidden_h = tf.zeros((1, 128))
        hidden_c = tf.zeros((1, 128))

        buffer.reset()
        done = False
        step_count = 0
        ep_reward = 0.0

        while not done and step_count < 1000:
            raw_action, value, log_prob, next_h, next_c = agent.get_action_and_value(
                obs_grid, telemetry_vec, prev_actions, hidden_h, hidden_c
            )

            # Step Environment
            scaled_actions = scale_actions_for_env(raw_action)
            transition = env.step(scaled_actions, step_count)

            # Store in Buffer
            buffer.store(
                obs_grid, telemetry_vec, prev_actions, raw_action, 
                transition.reward, value, log_prob, transition.done, 
                hidden_h.numpy(), hidden_c.numpy()
            )

            # State
            obs_grid = transition.obs_next_grid
            telemetry_vec = transition.obs_next_state.to_telemetry_vector()
            prev_actions = raw_action
            hidden_h = next_h
            hidden_c = next_c
            done = transition.done
            ep_reward += transition.reward
            step_count += 1

        if step_count < 10:
            buffer.reset()  # Dump useless data
            continue  # tart the next episode

        # End of Episode
        if step_count == 1000 and not done:
            _, last_value, _, _, _ = agent.get_action_and_value(
                obs_grid, telemetry_vec, prev_actions, hidden_h, hidden_c
            )
        else:
            last_value = 0.0

        advantages, returns = buffer.finish_path(last_value)

        # Run Gradient Update and retrieve losses
        actor_loss, critic_loss = agent.update_networks(buffer, advantages, returns, seq_len=16, epochs=3)

        # Tensorboard logging
        if actor_loss is not None:
            with summary_writer.as_default():
                tf.summary.scalar('Performance/Episode_Reward', ep_reward, step=ep)
                tf.summary.scalar('Performance/Steps_Survived', step_count, step=ep)
                tf.summary.scalar('Losses/Actor_Loss', actor_loss, step=ep)
                tf.summary.scalar('Losses/Critic_Loss', critic_loss, step=ep)

        # Calculate time remaining
        elapsed_time = time.time() - start_time
        time_left = max_duration_seconds - elapsed_time
        mins_left = int(time_left // 60)
        secs_left = int(time_left % 60)

        print(f"Ep {ep+1} | Steps: {step_count:3d} | Status: {transition.status:13s} | Reward: {ep_reward:6.2f} | Time Left: {mins_left:02d}:{secs_left:02d}")        

        # Save models every 50 episodes
        if (ep + 1) % 30 == 0:
            agent.save_models("saved_models/sailing_agent")

        ep += 1

    # Final save when time is up
    agent.save_models("saved_models/sailing_agent_final")
    print("--- Training Complete! ---")


if __name__ == "__main__":
    tf.config.run_functions_eagerly(False)
    run_test_training(duration_minutes=600)
