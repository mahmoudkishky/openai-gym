import tensorflow as tf
from tensorflow.keras.optimizers import Adam
import numpy as np
from models.lunar_lander import Agent
import random
from configs import lunar_lander as config
from collections import deque

class Trainer():

    def __init__(self, agent, env, batch_size, discount_rate, learning_rate, initial_epsilon, target_update_rate = config.DEFAULT_TARGET_UPDATE_RATE):

        self.agent = agent
        self.target_agent = Agent(config.ACTION_SPACE_DIM)
        self.target_agent.set_weights(self.agent.get_weights())
        self.env = env
        self.replay_buffer = deque(maxlen=config.MAX_REPLAY_BUFFER_SIZE)
        self.batch_size = batch_size
        self.discount_rate = discount_rate
        self.learning_rate = learning_rate
        self.epsilon = initial_epsilon
        self.target_update_rate = target_update_rate
        self.optimizer = Adam(learning_rate = learning_rate)
    
    def train(self):

        
        episode_rewards = []
        avg_reward = 0
        total_training_steps = 0

        for episode in range(1, config.MAX_EPISODE_COUNT):

            current_state, _ = self.env.reset()
            accumulated_episode_reward = 0
            
            for step in range(config.MAX_STEPS_PER_EPISODE):

                q_values = self.agent(np.expand_dims(current_state, axis=0))

                # epsilon greedy policy
                if random.random() > self.epsilon:
                    action = np.argmax(q_values.numpy()[0])
                else:
                    action = self.env.action_space.sample()

                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                accumulated_episode_reward = accumulated_episode_reward + reward
                # store in the replay buffer
                self.replay_buffer.append((current_state, action, reward, next_state, done))
                current_state = next_state

                # wait until we have enough samples in the replay buffer
                if len(self.replay_buffer) >= self.batch_size:
                    
                    # update agent weights
                    with tf.GradientTape() as gt:

                        # sample batch from replay buffer
                        batch = random.sample(self.replay_buffer, self.batch_size)
                        s, a, r, s_next, d = map(np.array, zip(*batch))

                        q_target = r + (1- d) * self.discount_rate * tf.reduce_max(self.target_agent(s_next), axis=1)
                        q_agent = tf.reduce_sum(self.agent(s) * tf.one_hot(a, config.ACTION_SPACE_DIM), axis=1)
                        
                        q_loss = tf.reduce_mean(tf.square(q_target - q_agent))
                        gradients = gt.gradient(q_loss, self.agent.trainable_variables)
                        valid_grads = [(g, v) for g, v in zip(gradients, self.agent.trainable_variables) if g is not None]
                        self.optimizer.apply_gradients(valid_grads)

                        total_training_steps = total_training_steps + 1

                    # update freezed target agent
                    if total_training_steps % self.target_update_rate == 0:
                        #print("Update target weights")
                        self.target_agent.set_weights(self.agent.get_weights())
                

                # end of simulation episode
                if terminated or truncated:
                        
                    # calculate averge accumulated reward
                    episode_rewards.append(accumulated_episode_reward)

                    # epsilon decay
                    self.epsilon = max(0.01, self.epsilon * 0.995)

                    if episode >= 100:
                        avg_reward = np.mean(episode_rewards[-100:])
                    else:
                        avg_reward = np.mean(episode_rewards)

                    if avg_reward >= 200:
                        print(f"The environment is solved in {episode} episodes")
                        self.agent.save_weights(f"weights/lunar_lander/solved_environment/dqn_agent")
                        return
            
                    if episode % 50 == 0:
                        self.agent.save_weights(f"weights/lunar_lander/episode_{episode}/dqn_agent")

                    print(f"Episode: {episode} | Steps: {step} | Total training steps = {total_training_steps} |Epsilon: {self.epsilon:.3f} | Average reward: {avg_reward:.2f}")
            
                    break
            
