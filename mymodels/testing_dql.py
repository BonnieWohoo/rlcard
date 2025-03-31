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
    "num_episodes": 1000,
    "num_eval_games": 20,
    "evaluate_every": 20,
    "log_dir": "experiments/uno_dqn_result/",
}


'''
 Args:
    replay_memory_size (int): Size of the replay memory
    replay_memory_init_size (int): Number of random experiences to sample when initializing
        the reply memory.
    update_target_estimator_every (int): Copy parameters from the Q estimator to the
        target estimator every N steps
    discount_factor (float): Gamma discount factor
    epsilon_start (float): Chance to sample a random action when taking an action.
        Epsilon is decayed over time and this is the start value
    epsilon_end (float): The final minimum value of epsilon after decaying is done
    epsilon_decay_steps (int): Number of steps to decay epsilon over
    batch_size (int): Size of batches to sample from the replay memory
    evaluate_every (int): Evaluate every N steps
    num_actions (int): The number of the actions
    state_space (list): The space of the state vector
    train_every (int): Train the network every X steps.
    mlp_layers (list): The layer number and the dimension of each layer in MLP
    learning_rate (float): The learning rate of the DQN agent.
    device (torch.device): whether to use the cpu or gpu
    save_path (str): The path to save the model checkpoints
    save_every (int): Save the model every X training steps
    '''

ARGS = {
    "replay_memory_size":20000,
    "replay_memory_init_size":100,
    "update_target_estimator_every":1000,
    "discount_factor":0.99,
    "epsilon_start":1.0,
    "epsilon_end":0.1,
    "epsilon_decay_steps":20000,
    "batch_size":32,
    "num_actions":2,
    "state_shape":None,
    "train_every":1,
    "mlp_layers":None,
    "learning_rate":0.00005,
    "device":None,
    "save_path":None,
    "save_every":float('inf'),
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
        logger.log("=========CONFIG=========")
        for key, value in CONFIG.items():
            text = f"{key}: {value}"
            logger.log(text)
        logger.log("==========ARGS==========")
        for key, value in ARGS.items():
            text = f"{key}: {value}"
            logger.log(text)

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