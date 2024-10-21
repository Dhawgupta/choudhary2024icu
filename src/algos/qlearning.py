import argparse
import os, sys, time
sys.path.append(os.getcwd())
import random
import time
from distutils.util import strtobool

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from src.utils.utils import set_seeds, make_env, get_mask, encode_state, ReplayBuffer, Transition
from src.utils.models import QNetwork, linear_schedule



def run_qlearning(args, use_tensorboard=False, use_wandb=False):
    assert args.num_envs == 1, "vectorized envs are not supported at the moment"
    returns_ = -1 * np.ones([args.max_episodes])
    discounted_returns_ = -1 *  np.ones([args.max_episodes])
    num_steps = np.zeros([args.max_episodes])

    run_name = f"qlearning_{args.seed}_{int(time.time())}"
    if use_wandb:
        import wandb 
        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
    if use_tensorboard:
        writer = SummaryWriter(log_dir=f"runs/{run_name}")
        writer.add_text(
            "hyperparameters",
            "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
        )


    set_seeds(args.seed)
    device = torch.device("cpu")


        # env setup
    if hasattr(args, 'env_type'):
        env_type = int(args.env_type)
    else:
        env_type = None
    envs = gym.vector.SyncVectorEnv(
            [make_env( args.seed + i, env_type) for i in range(args.num_envs)]
        )

    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"

    q_network = QNetwork(envs).to(device)
    optimizer = optim.Adam(q_network.parameters(), lr=args.learning_rate)
    n_states = envs.single_observation_space.n
    n_actions = envs.single_action_space.n
    print(f"Number of States: {n_states}, Number of actions: {n_actions}")

    rb = ReplayBuffer(
        args.buffer_size
    )

    start_time = time.time()

    # TRY NOT TO MODIFY: start the game
    states, infos = envs.reset(seed=args.seed)
    obs = encode_state(states, n_states)
    action_masks = get_mask(infos, args.num_envs, n_actions)
    start_index = 0
    episode_number = 0
    global_step = 0
    while episode_number < args.max_episodes:
        # ALGO LOGIC: put action logic here
        epsilon = linear_schedule(args.start_e, args.end_e, args.exploration_fraction * args.max_episodes, episode_number)
        allowed_actions = infos['admissible_actions'][0]
        if random.random() < epsilon:
            actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
            # pick a random action from allowed_action list
            actions = np.array([random.choice(allowed_actions) for _ in range(envs.num_envs)])
        else:
            q_values = q_network(obs.to(device), action_masks.to(device))
            actions = torch.argmax(q_values, dim=1).cpu().numpy()

        # TRY NOT TO MODIFY: execute the game and log data.
        next_states, rewards, terminated, truncated, infos = envs.step(actions)
        global_step += 1
        next_obs = encode_state(next_states, n_states)
        next_action_masks = get_mask(infos, args.num_envs, n_actions)

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        if terminated:
            # print(start_index, global_step)
            total_reward = infos['final_info'][0]['episode']['r']
            episode_length = infos['final_info'][0]['episode']['l']
            total_return = total_reward * (args.gamma**(episode_length-1)) # we only have reward at the end
            returns_[episode_number] = total_reward
            discounted_returns_[episode_number] = total_return
            num_steps[episode_number] = episode_length
            
            episode_number += 1
            
            # episode_numbers[start_index:global_step] = episode_number
            start_index = global_step
            if use_tensorboard:
                writer.add_scalar("charts/episodic_return", total_reward, episode_number)
                writer.add_scalar("charts/episodic_length", episode_length, episode_number)
                writer.add_scalar("charts/episodic_discounted_return", total_return, episode_number)
                writer.add_scalar("charts/episodic_number", episode_number, episode_number)
                writer.add_scalar("charts/num_steps", global_step, episode_number)
                writer.add_scalar("charts/epsilon", epsilon, episode_number)

        obs = torch.Tensor(obs).reshape((1,-1)).float()
        next_obs = torch.Tensor(next_obs).reshape((1,-1)).float()
        actions = torch.Tensor(actions).reshape((1,-1)).long()
        rewards = torch.Tensor(rewards).reshape((1,-1)).float()
        next_action_masks = torch.Tensor(next_action_masks).reshape((1,-1)).float()
        terminated = torch.Tensor(terminated).reshape((1,-1)).float()
        data = Transition(obs, actions, action_masks, rewards, next_obs, next_action_masks, terminated)

        obs = next_obs
        action_masks = next_action_masks


        with torch.no_grad():
            target_max, _ = q_network(data.next_state, data.next_action_mask).max(dim=1)
            td_target = data.reward.flatten() + args.gamma * target_max * (1 - data.done.flatten())
        old_val = q_network(data.state).gather(1, data.action).squeeze()
        loss = F.mse_loss(td_target, old_val)

        if global_step % 1000 == 0:
            if use_tensorboard:
                writer.add_scalar("losses/td_loss", loss, global_step)
                writer.add_scalar("losses/q_values", old_val.mean().item(), global_step)
                writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)
            print("Mean Q Value:", old_val.mean().item())
            print(f"SPS: {int(global_step / (time.time() - start_time))}, Episode: {int(episode_number)}, Step: {global_step}, Return: {total_reward}, Episode length: {episode_length}")
            print("Steps:", global_step)
                
        # optimize the model
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    
    
    assert (returns_ >= 0.0).all(), "returns should be non-negative"
    assert (discounted_returns_ >= 0.0).all(), "discounted returns should be non-negative"
    envs.close()
    if use_tensorboard: writer.close()
    return returns_, discounted_returns_, num_steps

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, default=os.path.basename(__file__).rstrip(".py"),
        help="the name of this experiment")
    parser.add_argument("--seed", type=int, default=0,
        help="seed of the experiment")


    # Algorithm specific arguments
    parser.add_argument("--max-episodes", type=int, default=1000)
    parser.add_argument("--learning-rate", type=float, default=2.5e-4,
        help="the learning rate of the optimizer")
    parser.add_argument("--num-envs", type=int, default=1,
        help="the number of parallel game environments")
    parser.add_argument("--buffer-size", type=int, default=1,
        help="the replay memory buffer size")
    parser.add_argument("--gamma", type=float, default=0.99,
        help="the discount factor gamma")
    parser.add_argument("--batch-size", type=int, default=1,
        help="the batch size of sample from the reply memory")
    parser.add_argument("--start-e", type=float, default=1,
        help="the starting epsilon for exploration")
    parser.add_argument("--end-e", type=float, default=0.05,
        help="the ending epsilon for exploration")
    parser.add_argument("--exploration-fraction", type=float, default=0.5,
        help="the fraction of `total-timesteps` it takes from start-e to go end-e")
    parser.add_argument("--track", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="if toggled, this experiment will be tracked with Weights and Biases")

    args = parser.parse_args()
    assert args.num_envs == 1, "vectorized envs are not supported at the moment"
    
    run_qlearning(args, use_tensorboard=True, use_wandb=args.track)
