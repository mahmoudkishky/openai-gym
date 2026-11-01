import tensorflow as tf
from tensorflow.keras.optimizers import Adam
import numpy as np
from models.walker import Actor, Critic
from configs import walker as config
from replay_buffer import ReplayBuffer
import os

class Trainer():

    def __init__(self, actor, env, batch_size, discount_rate, learning_rate_actor, learning_rate_critic, tau=0.005, alpha=0.2):

        self.actor = actor
        self.critic1 = Critic()
        self.critic2 = Critic()

        # Target critics
        self.critic1_target = Critic()
        self.critic2_target = Critic()
        self.critic1_target.set_weights(self.critic1.get_weights())
        self.critic2_target.set_weights(self.critic2.get_weights())

        self.env = env
        self.replay_buffer = ReplayBuffer()
        self.batch_size = batch_size
        self.discount_rate = discount_rate
        self.log_alpha = tf.Variable(np.log(alpha), dtype=tf.float32, trainable=True)
        self.tau = tau

        self.actor_optimizer = Adam(learning_rate_actor)
        self.critic_optimizer = Adam(learning_rate_critic)
        self.alpha_optimizer = Adam(3e-4)
        self.target_entropy = -3

        self.log_dir = "logs"
        self.summary_writer = tf.summary.create_file_writer(self.log_dir)

    def compute_log_prob(self, mu, std, a):

        """Compute log π(a|s) for tanh-squashed Gaussian (actor returns std)."""
        EPS = 1e-6
        # Invert tanh
        u = tf.atanh(tf.clip_by_value(a, -1 + EPS, 1 - EPS))
        log_prob_u = -0.5 * (((u - mu) / (std + EPS)) ** 2 + 2 * tf.math.log(std + EPS) + np.log(2 * np.pi))
        log_prob_u = tf.reduce_sum(log_prob_u, axis=-1, keepdims=True)
        log_prob = log_prob_u - tf.reduce_sum(tf.math.log(1 - tf.square(a) + EPS), axis=-1, keepdims=True)
        return log_prob


    def soft_update(self, target, source):

        for t, s in zip(target.variables, source.variables):
            t.assign((1 - self.tau) * t + self.tau * s)

    def train(self):

        episode_rewards = []
        avg_reward = 0
        total_training_steps = 0

        for episode in range(1, config.MAX_EPISODE_COUNT + 1):

            current_state, _ = self.env.reset()
            accumulated_episode_reward = 0

            for step in range(config.MAX_STEPS_PER_EPISODE):
                
                # sample action using the Gaussian assumption
                mu, std = self.actor(tf.expand_dims(current_state, axis=0))
                action = tf.tanh(mu + std * tf.random.normal(shape=mu.shape))
                action = np.array(action, dtype=np.float32).reshape(self.env.action_space.shape)

                # run one simulation step and retrieve transaction, store to the replay buffer
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                accumulated_episode_reward += reward

                self.replay_buffer.add_item((current_state, action, reward, next_state, done))
                current_state = next_state
                
                critic1_loss = critic2_loss = alpha_loss = actor_loss = 0

                if self.replay_buffer.get_size() >= 10000:
                    
                    # sample minibatch from the replay buffer
                    batch = self.replay_buffer.sample(self.batch_size)
                    s, a, r, s_next, d = map(np.array, zip(*batch))
                    
                    s = tf.convert_to_tensor(s, dtype=tf.float32)
                    a = tf.convert_to_tensor(a, dtype=tf.float32)
                    s_next = tf.convert_to_tensor(s_next, dtype=tf.float32)
                    r = tf.expand_dims(tf.convert_to_tensor(r, dtype=tf.float32), axis=-1)
                    d = tf.expand_dims(tf.convert_to_tensor(d, dtype=tf.float32), axis=-1)

                    if total_training_steps == 0:
                        print("==========================================")
                        print(f"state shape (s): {s.shape}")
                        print(f"action shape (a): {a.shape}")
                        print(f"next_state shape (s_next): {s_next.shape}")
                        print(f"reward shape (r): {r.shape}")
                        print(f"done shape (d): {d.shape}")
                        mu_test, std_test = self.actor(s)
                        print(f"mu shape: {mu_test.shape}, std shape: {std_test.shape}")
                        print(f"critic output shape (q1): {self.critic1(s, a).shape}")
                        print("==========================================\n")

                    # compute pi(at+1|st+1)
                    mu_next, std_next = self.actor(s_next)
                    a_next = tf.tanh(mu_next + std_next * tf.random.normal(shape=mu_next.shape))

                    # interpret entropy as log pi(a+1|s+1)
                    entropy_critic = -self.compute_log_prob(mu_next, std_next, a_next)

                    # compute Q1(st+1, at+1) and Q2(st+1, at+1)
                    q1_target = self.critic1_target(s_next, a_next)
                    q2_target = self.critic2_target(s_next, a_next)

                    min_q_target = tf.minimum(q1_target, q2_target)

                    # Bellman target for critics
                    y = r + (1 - d) * self.discount_rate * (min_q_target - tf.clip_by_value(tf.exp(self.log_alpha), 1e-2, 1.0) * entropy_critic)
                    y = tf.stop_gradient(y)


                    # update critics
                    with tf.GradientTape(persistent=True) as tape:

                        q1 = self.critic1(s, a)
                        q2 = self.critic2(s, a)
                        critic1_loss = tf.reduce_mean(tf.square(q1 - y))
                        critic2_loss = tf.reduce_mean(tf.square(q2 - y))

                    grads_c1 = tape.gradient(critic1_loss, self.critic1.trainable_variables)
                    grads_c2 = tape.gradient(critic2_loss, self.critic2.trainable_variables)

                    self.critic_optimizer.apply_gradients(zip(grads_c1, self.critic1.trainable_variables))
                    self.critic_optimizer.apply_gradients(zip(grads_c2, self.critic2.trainable_variables))

                    del tape

                    # update actor
                    with tf.GradientTape() as tape:

                        mu_a, std_a = self.actor(s)
                        a_pred = tf.tanh(mu_a + std_a * tf.random.normal(shape=mu_a.shape))

                        entropy_actor = -self.compute_log_prob(mu_a, std_a, a_pred)

                        q1_pi = self.critic1(s, a_pred)
                        q2_pi = self.critic2(s, a_pred)

                        min_q_pi = tf.minimum(q1_pi, q2_pi)
                        actor_loss = -tf.reduce_mean(min_q_pi - tf.clip_by_value(tf.exp(self.log_alpha), 1e-2, 1.0) * entropy_actor)

                    actor_grads = tape.gradient(actor_loss, self.actor.trainable_variables)
                    self.actor_optimizer.apply_gradients(zip(actor_grads, self.actor.trainable_variables))

                    self.soft_update(self.critic1_target, self.critic1)
                    self.soft_update(self.critic2_target, self.critic2)
                    
                    del tape

                    with tf.GradientTape() as tape:
                        alpha_loss = -tf.reduce_mean(self.log_alpha * tf.stop_gradient(entropy_actor + self.target_entropy))

                    alpha_grads = tape.gradient(alpha_loss, [self.log_alpha])
                    self.alpha_optimizer.apply_gradients(zip(alpha_grads, [self.log_alpha]))


                    total_training_steps += 1

                    with self.summary_writer.as_default():
                        tf.summary.scalar('Reward/Average', np.mean(episode_rewards[-100:]), step=episode)
                        tf.summary.scalar('Loss/Actor', actor_loss, step=step)
                        tf.summary.scalar('Loss/Critic1', critic1_loss, step=step)
                        tf.summary.scalar('Loss/Critic2', critic2_loss, step=step)
                        tf.summary.scalar('Loss/Alpha', alpha_loss, step=step)

                    if total_training_steps % 500 == 0:
                        print(f"Actor loss: {actor_loss:.2f} | Critic 1 loss: {critic1_loss:.2f} | Critic 2 loss: {critic2_loss:.2f}")


                        
                if done:

                    episode_rewards.append(accumulated_episode_reward)
                    avg_reward = np.mean(episode_rewards[-100:]) if len(episode_rewards) > 100 else np.mean(episode_rewards)

                    if avg_reward >= 2500:
                            
                        print("Environment is solved, save model")
                        os.makedirs("weights/walker", exist_ok=True)
                        self.actor.save_weights("weights/walker/actor_weights")
                        return

                    if episode % 50 == 0:
                        print(f"Episode: {episode} | Steps: {step} | Reward: {reward:.2f} | Avg Reward: {avg_reward:.2f} | alpha: {tf.exp(self.log_alpha)}")
                    
                    self.env.reset()
                    break
                        
        self.summary_writer.close()