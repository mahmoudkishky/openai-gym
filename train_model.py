import argparse
import importlib

import gymnasium as gym

ENVS = {
    "inverted_pendulum": "InvertedPendulum-v4",
    "lunar_lander": "LunarLander-v2",
    "walker": "Walker2d-v4",
}


def build_trainer(name, env, config):
    models = importlib.import_module(f"models.{name}")
    trainers = importlib.import_module(f"trainers.{name}")

    if name == "inverted_pendulum":
        agent = models.Agent(config.ACTION_SPACE_DIM)
        return trainers.Trainer(agent, env, config.DISCOUNT_RATE, config.LEARNING_RATE)
    if name == "lunar_lander":
        agent = models.Agent(config.ACTION_SPACE_DIM)
        return trainers.Trainer(agent, env, config.BATCH_SIZE, config.DISCOUNT_RATE,
                                config.LEARNING_RATE, config.EPSILON)
    actor = models.Actor(config.ACTION_SPACE_DIM)
    return trainers.Trainer(actor, env, config.BATCH_SIZE, config.DISCOUNT_RATE,
                            config.LEARNING_RATE_ACTOR, config.LEARNING_RATE_CRITIC, config.SOFT_UPDATE)


def main(args):
    config = importlib.import_module(f"configs.{args.env}")
    env = gym.make(ENVS[args.env], render_mode=args.render_mode,
                   max_episode_steps=config.MAX_STEPS_PER_EPISODE)
    build_trainer(args.env, env, config).train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", choices=ENVS, required=True, help="environment to train")
    parser.add_argument("--render-mode", dest="render_mode", default="human",
                        help="gymnasium render mode (e.g. human, rgb_array)")
    main(parser.parse_args())
