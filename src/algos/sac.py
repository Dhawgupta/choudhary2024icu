# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/sac/#sac_ataripy
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

from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter


from src.utils.utils import make_env, many_hot, get_mask, ReplayBuffer  
from src.utils.utils import encode_state, set_seeds
from src.utils.models import ActorCritic, QNetwork





def run_sac(args, use_tensorboard=False, use_wandb=False):
    assert args.num_envs == 1, "vectorized envs are not supported at the moment"
    returns_ = -1 * np.ones([args.max_episodes])
    discounted_returns_ = -1 *  np.ones([args.max_episodes])
    num_steps = np.zeros([args.max_episodes])

    run_name = f"sac_{args.seed}_{int(time.time())}"
    if use_wandb:
        import wandb 
        wandb.init(
            # project=args.wandb_project_name,
            project='sepsis',
            # entity=args.wandb_entity,
            sync_tensorboard=False,
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
    # TRY NOT TO MODIFY: seeding
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


    actor = ActorCritic(envs).to(device)
    qf1 = QNetwork(envs).to(device)
    qf2 = QNetwork(envs).to(device)
    qf1_target = QNetwork(envs).to(device)
    qf2_target = QNetwork(envs).to(device)
    qf1_target.load_state_dict(qf1.state_dict())
    qf2_target.load_state_dict(qf2.state_dict())
    # TRY NOT TO MODIFY: eps=1e-4 increases numerical stability
    q_optimizer = optim.Adam(list(qf1.parameters()) + list(qf2.parameters()), lr=args.q_lr, eps=1e-4)
    actor_optimizer = optim.Adam(list(actor.parameters()), lr=args.policy_lr, eps=1e-4)

    # Automatic entropy tuning
    if args.autotune:
        target_entropy = -args.target_entropy_scale * torch.log(1 / torch.tensor(envs.single_action_space.n))
        log_alpha = torch.zeros(1, requires_grad=True, device=device)
        alpha = log_alpha.exp().item()
        a_optimizer = optim.Adam([log_alpha], lr=args.q_lr, eps=1e-4)
    else:
        alpha = args.alpha

    rb = ReplayBuffer(
        args.buffer_size
    )
    n_states = envs.single_observation_space.n
    n_actions = envs.single_action_space.n
    print(f"Number of States: {n_states}, Number of actions: {n_actions}")
    start_time = time.time()

    # TRY NOT TO MODIFY: start the game
    states, infos  = envs.reset()
    obs = encode_state(states, n_states)
    action_masks = get_mask(infos, args.num_envs, n_actions)
    episode_number = 0
    start_index = 0
    global_step = 0
    while episode_number < args.max_episodes:
        # ALGO LOGIC: put action logic here
        allowed_actions = infos['admissible_actions'][0]
        
        if global_step < args.learning_starts:
            # actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
            actions = np.array([random.choice(allowed_actions) for _ in range(envs.num_envs)])
        else:
            actions, _, _= actor.get_action(torch.Tensor(obs).to(device), action_mask = action_masks)
            actions = actions.detach().cpu().numpy()

        # TRY NOT TO MODIFY: execute the game and log data.
        next_states, rewards, terminated, truncated, infos = envs.step(actions)
        next_obs = encode_state(next_states, n_states)
        next_action_masks = get_mask(infos, args.num_envs, n_actions)
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

        # rb.add(obs, real_next_obs, actions, rewards, dones, infos)
        rb.push(obs, actions, action_masks, rewards, next_obs, next_action_masks, terminated)
        # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
        obs = next_obs
        action_masks = next_action_masks

        # ALGO LOGIC: training.
        if global_step > args.learning_starts:
            if global_step % args.update_frequency == 0:
                data = rb.sample(args.batch_size)
                # CRITIC training
                with torch.no_grad():
                    # TODO check for next action masks
                    _, next_state_log_pi, next_state_action_probs = actor.get_action(data.next_state)
                    qf1_next_target = qf1_target(data.next_state)
                    qf2_next_target = qf2_target(data.next_state)
                    # we can use the action probabilities instead of MC sampling to estimate the expectation
                    min_qf_next_target = next_state_action_probs * (
                        torch.min(qf1_next_target, qf2_next_target) - alpha * next_state_log_pi
                    )
                    # adapt Q-target for discrete Q-function
                    min_qf_next_target = min_qf_next_target.sum(dim=1)
                    next_q_value = data.reward.flatten() + (1 - data.done.flatten()) * args.gamma * (min_qf_next_target)

                # use Q-values only for the taken actions
                qf1_values = qf1(data.state)
                qf2_values = qf2(data.state)
                qf1_a_values = qf1_values.gather(1, data.action.long()).view(-1)
                qf2_a_values = qf2_values.gather(1, data.action.long()).view(-1)
                qf1_loss = F.mse_loss(qf1_a_values, next_q_value)
                qf2_loss = F.mse_loss(qf2_a_values, next_q_value)
                qf_loss = qf1_loss + qf2_loss

                q_optimizer.zero_grad()
                qf_loss.backward()
                q_optimizer.step()

                # ACTOR training
                _, log_pi, action_probs = actor.get_action(data.state)
                with torch.no_grad():
                    qf1_values = qf1(data.state)
                    qf2_values = qf2(data.state)
                    min_qf_values = torch.min(qf1_values, qf2_values)
                # no need for reparameterization, the expectation can be calculated for discrete actions
                actor_loss = (action_probs * ((alpha * log_pi) - min_qf_values)).mean()

                actor_optimizer.zero_grad()
                actor_loss.backward()
                actor_optimizer.step()

                if args.autotune:
                    # re-use action probabilities for temperature loss
                    alpha_loss = (action_probs.detach() * (-log_alpha.exp() * (log_pi + target_entropy).detach())).mean()

                    a_optimizer.zero_grad()
                    alpha_loss.backward()
                    a_optimizer.step()
                    alpha = log_alpha.exp().item()

            # update the target networks
            if global_step % args.target_network_frequency == 0:
                for param, target_param in zip(qf1.parameters(), qf1_target.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)
                for param, target_param in zip(qf2.parameters(), qf2_target.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)

            if global_step % 100 == 0:
                print(f"SPS: {int(global_step / (time.time() - start_time))}, Episode: {int(episode_number)}, Step: {global_step}, Return: {total_reward}, Episode length: {episode_length}")
                if use_tensorboard:
                    writer.add_scalar("losses/qf1_values", qf1_a_values.mean().item(), global_step)
                    writer.add_scalar("losses/qf2_values", qf2_a_values.mean().item(), global_step)
                    writer.add_scalar("losses/qf1_loss", qf1_loss.item(), global_step)
                    writer.add_scalar("losses/qf2_loss", qf2_loss.item(), global_step)
                    writer.add_scalar("losses/qf_loss", qf_loss.item() / 2.0, global_step)
                    writer.add_scalar("losses/actor_loss", actor_loss.item(), global_step)
                    writer.add_scalar("losses/alpha", alpha, global_step)
                    # print("SPS:", int(global_step / (time.time() - start_time)))
                    writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)
                    if args.autotune:
                        writer.add_scalar("losses/alpha_loss", alpha_loss.item(), global_step)

        global_step += 1
    

    assert (returns_ >= 0.0).all(), "returns should be non-negative"
    assert (discounted_returns_ >= 0.0).all(), "discounted returns should be non-negative"
    envs.close()
    if use_tensorboard: writer.close()

    return returns_, discounted_returns_, num_steps


if __name__ == '__main__':
        # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0, 
        help="seed of the experiment")
    # parser.add_argument("--total-timesteps", type=int, default=50000,
    #     help="total timesteps of the experiments")
    parser.add_argument("--max-episodes", type=int, default=1000, help="number of episodes of the evaluation")
    parser.add_argument("--buffer-size", type=int, default=int(1e6),
        help="the replay memory buffer size") # smaller than in original paper but evaluation is done only for 100k steps anyway
    parser.add_argument("--gamma", type=float, default=0.99,
        help="the discount factor gamma")
    parser.add_argument("--tau", type=float, default=1.0,
        help="target smoothing coefficient (default: 1)") # Default is 1 to perform replacement update
    parser.add_argument("--batch-size", type=int, default=64,
        help="the batch size of sample from the reply memory")
    parser.add_argument("--learning-starts", type=int, default=2e4,
        help="timestep to start learning")
    parser.add_argument("--policy-lr", type=float, default=3e-4,
        help="the learning rate of the policy network optimizer")
    parser.add_argument("--q-lr", type=float, default=3e-4,
        help="the learning rate of the Q network network optimizer")
    parser.add_argument("--update-frequency", type=int, default=4,
        help="the frequency of training updates")
    parser.add_argument("--target-network-frequency", type=int, default=8000,
        help="the frequency of updates for the target networks")
    parser.add_argument("--alpha", type=float, default=0.2,
        help="Entropy regularization coefficient.")
    parser.add_argument("--autotune", type=lambda x:bool(strtobool(x)), default=True, nargs="?", const=True,
        help="automatic tuning of the entropy coefficient")
    parser.add_argument("--target-entropy-scale", type=float, default=0.89,
        help="coefficient for scaling the autotune entropy target")
    parser.add_argument("--num-envs", type=int, default=1,
        help="the number of parallel game environments")
    parser.add_argument("--track", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="if toggled, this experiment will be tracked with Weights and Biases")
    args = parser.parse_args()
    assert args.num_envs == 1, "vectorized envs are not supported at the moment"
    
    run_sac(args, use_tensorboard=True, use_wandb=args.track)