import os
import argparse

import torch

import rlcard
from rlcard.agents import RandomAgent
from rlcard.utils import (
    get_device,
    set_seed,
    tournament,
    reorganize,
    Logger,
    plot_curve,
)

# ==== Fixed Parameters ====
CONFIG = {
    "env": "uno",  # Change to "uno" or any other game
    "algorithm": "dqn",
    "seed": 42,
    "num_episodes": 100,
    "num_eval_games": 20,
    "evaluate_every": 20,
    "log_dir": "experiments/uno_dqn_result/",
}


def train():

    # Check whether gpu is available
    device = get_device()

    # Seed numpy, torch, random
    set_seed(CONFIG["seed"])

    # Make the environment with seed
    env = rlcard.make(
        CONFIG["env"],
        config={
            'seed': CONFIG["seed"],
        }
    )

    # Initialize the agent and use random agents as opponents
    from rlcard.agents import DQNAgent
    agent = DQNAgent(
        num_actions=env.num_actions,
        state_shape=env.state_shape[0],
        mlp_layers=[64,64],
        device=device,
    )

    agents = [agent]
    for _ in range(1, env.num_players):
        agents.append(RandomAgent(num_actions=env.num_actions))
    env.set_agents(agents)

    # Start training
    with Logger(CONFIG["log_dir"]) as logger:
        for episode in range(CONFIG["num_episodes"]):

            # Generate data from the environment
            trajectories, payoffs = env.run(is_training=True)

            # Reorganaize the data to be state, action, reward, next_state, done
            trajectories = reorganize(trajectories, payoffs)

            # Feed transitions into agent memory, and train the agent
            # Here, we assume that DQN always plays the first position
            # and the other players play randomly (if any)
            for ts in trajectories[0]:
                agent.feed(ts)

            # Evaluate the performance. Play with random agents.
            if episode % CONFIG["evaluate_every"] == 0:
                logger.log_performance(
                    episode,
                    tournament(
                        env,
                        CONFIG["num_eval_games"],
                    )[0]
                )

        # Get the paths
        csv_path, fig_path = logger.csv_path, logger.fig_path

    # Plot the learning curve
    plot_curve(csv_path, fig_path, CONFIG["algorithm"])

    # Save model
    save_path = os.path.join(CONFIG["log_dir"], 'model.pth')
    torch.save(agent, save_path)
    print('Model saved in', save_path)

train()