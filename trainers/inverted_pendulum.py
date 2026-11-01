import tensorflow as tf
from tensorflow.keras.optimizers import Adam
import numpy as np
from models.inverted_pendulum import Agent
import random
from configs import inverted_pendulum as config
from collections import deque

class Trainer():

    def __init__(self, agent, env, discount_rate, learning_rate):

        self.agent = agent
        self.env = env
        self.discount_rate = discount_rate
        self.optimizer = Adam(learning_rate)
    
    def compute_discounted_rewards(self, rewards):
        
        discounted_rewards = []
        R = 0

        for r in reversed(rewards):
            R = r + self.discount_rate * R
            discounted_rewards.insert(0, R)

        return discounted_rewards

    
    def train(self):

        
        episode_rewards = deque(maxlen=100)
        total_training_steps = 0
        avg_reward = 0

        for episode in range(1, config.MAX_EPISODE_COUNT):

            current_state, _ = self.env.reset()
            accumulated_episode_reward = 0
            actions = []
            rewards = []
            states = []
            done = False

            # collect the trajectory
            while not done:
                
                # sample the actions from our model (Gaussian assumption)
                mu, std = self.agent(np.expand_dims(current_state, axis=0))
                action =  mu + std * tf.random.normal(shape=mu.shape)
                action = np.array(action, dtype=np.float32).reshape(self.env.action_space.shape)

                # compute reward
                next_s, r, terminated, truncated, _ = self.env.step(config.ACTION_SCALE * action)

                # record the transitions
                actions.append(action)
                states.append(current_state)
                rewards.append(r)
                current_state = next_s
                accumulated_episode_reward = accumulated_episode_reward + r
                total_training_steps = total_training_steps + 1
                done = terminated or truncated
  

            # update the policy
            with tf.GradientTape() as gt:

                # predict actions means
                mu, std = self.agent(np.array(states))
                discounted_returns = self.compute_discounted_rewards(rewards)

                # return normalization
                discounted_returns = (discounted_returns - np.mean(discounted_returns)) / (np.std(discounted_returns) + 1e-8)

                policy_log = -0.5 * tf.reduce_sum(((np.array(actions) - mu) ** 2) / (std ** 2) + tf.math.log(2.0 * np.pi * std ** 2), axis=1)
                loss_entropy = 0.5 * tf.reduce_sum(tf.math.log(2 * np.pi * std ** 2) + 1)

                # the negative sign for gradient ascend
                policy_loss = -tf.reduce_mean(policy_log * np.array(discounted_returns) + 0.001 * loss_entropy)
                gradients = gt.gradient(policy_loss, self.agent.trainable_variables)

                # gradients clipping and checking
                gradients = [tf.clip_by_norm(g, 1.0) for g in gradients if g is not None]
                valid_grads = [(g, v) for g, v in zip(gradients, self.agent.trainable_variables) if g is not None]

                self.optimizer.apply_gradients(valid_grads)


            # logging                
            episode_rewards.append(accumulated_episode_reward)
            avg_reward = np.mean(episode_rewards)

            if avg_reward >= 500:
                print(f"The environment is solved in {episode} episodes")
                self.agent.save_weights(f"weights/inverted_pendulum/agent_pg")
                self.env.close()
                return
            
            if episode % 50 == 0:
                print(f"Episode: {episode} | Total training steps: {total_training_steps} | Episode reward: {accumulated_episode_reward} | Average reward: {avg_reward:.2f}")
                        
