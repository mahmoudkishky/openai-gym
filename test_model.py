import argparse
import importlib
import os

import gymnasium as gym
import numpy as np
import tensorflow as tf

ENVS = {
    "inverted_pendulum": "InvertedPendulum-v4",
    "lunar_lander": "LunarLander-v2",
    "walker": "Walker2d-v4",
}


def test_inverted_pendulum(args, config):
    from models.inverted_pendulum import Agent

    env = gym.make(ENVS["inverted_pendulum"], render_mode="human", max_episode_steps=10000)
    agent = Agent(config.ACTION_SPACE_DIM)

    path = args.weights or "weights/inverted_pendulum/agent_pg"
    if os.path.isdir(path):
        path = tf.train.latest_checkpoint(path) or os.path.join(path, "agent_pg")
    agent.load_weights(path)

    state, _ = env.reset()
    while True:
        mu, _ = agent(np.expand_dims(state, axis=0))
        action = np.array(mu, dtype=np.float32).reshape(env.action_space.shape)
        state, _, terminated, truncated, _ = env.step(config.ACTION_SCALE * action)
        if terminated or truncated:
            env.close()
            break


def test_lunar_lander(args, config):
    from models.lunar_lander import Agent

    env = gym.make(ENVS["lunar_lander"], render_mode="human")
    agent = Agent(config.ACTION_SPACE_DIM)
    path = args.weights or f"weights/lunar_lander/episode_{args.step}/dqn_agent"
    if os.path.isdir(path):
        path = tf.train.latest_checkpoint(path) or os.path.join(path, "dqn_agent")
    agent.load_weights(path)

    total_reward = 0
    success = 0
    for _ in range(args.episodes):
        state, _ = env.reset()
        for _ in range(1000):
            q_values = agent(np.expand_dims(state, axis=0))
            action = np.argmax(q_values.numpy()[0])
            state, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward

            if terminated or truncated:
                if reward >= 100:
                    print("Successful landing")
                    success += 1
                else:
                    print("Failed landing")
                break

    print(f"Average success rate = {100 * (success / args.episodes):4f}% "
          f"with average reward = {total_reward / args.episodes:4f}")


def test_walker(args, config):
    from models.walker import Actor

    env = gym.make(ENVS["walker"], render_mode="human", max_episode_steps=5000)
    actor = Actor(action_dim=env.action_space.shape[0])
    state, _ = env.reset()
    actor(tf.expand_dims(state, axis=0))  # build variables before loading

    path = args.weights or "weights/walker/actor_weights"
    if os.path.isdir(path):
        path = tf.train.latest_checkpoint(path) or os.path.join(path, "actor_weights")
    actor.load_weights(path)

    while True:
        mu, _ = actor(tf.expand_dims(state, axis=0))
        action = np.array(tf.tanh(mu)[0], dtype=np.float32)
        state, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            state, _ = env.reset()


TESTERS = {
    "inverted_pendulum": test_inverted_pendulum,
    "lunar_lander": test_lunar_lander,
    "walker": test_walker,
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", choices=ENVS, required=True, help="environment to test")
    parser.add_argument("--weights", default=None, help="weights path (overrides the default for the env)")
    parser.add_argument("--step", type=int, default=100000, help="lunar_lander: episode checkpoint to load")
    parser.add_argument("--episodes", type=int, default=10, help="lunar_lander: number of test episodes")
    args = parser.parse_args()
    config = importlib.import_module(f"configs.{args.env}")
    TESTERS[args.env](args, config)
