import os,sys
sys.path.append(os.getcwd())
import time
import torch
import random
import argparse
import numpy as np
import torch.nn as nn
import gymnasium as gym
import torch.optim as optim
from distutils.util import strtobool
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter

# custom classes
from src.utils.utils import make_env, many_hot, get_mask 
from src.utils.utils import encode_state, set_seeds
from src.utils.models import ActorCritic as Agent



    
def run_ppo(args, use_tensorboard=False, use_wandb=False):    
    assert args.num_envs == 1, "vectorized envs are not supported at the moment" # for purpose of fair evaluation
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    # make objects for collecting data

    # RECORD FOR LOGGING
    
    returns_ = -1 * np.ones([args.max_episodes])
    discounted_returns_ = -1 *  np.ones([args.max_episodes])
    num_steps = np.zeros([args.max_episodes])
    run_name = f"ppo_{args.seed}_{int(time.time())}"
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


        # env setup
    if hasattr(args, 'env_type'):
        env_type = int(args.env_type)
    else:
        env_type = None
    envs = gym.vector.SyncVectorEnv(
            [make_env( args.seed + i, env_type) for i in range(args.num_envs)]
        )

    # define local buffer
    agent = Agent(envs)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)
    n_states = envs.single_observation_space.n
    n_actions = envs.single_action_space.n
    print(f"Number of States: {n_states}, Number of actions: {n_actions}")
    obs = torch.zeros((args.num_steps, args.num_envs) + (n_states,))
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape)
    action_masks = torch.zeros((args.num_steps, args.num_envs) + (n_actions,))
    logprobs = torch.zeros((args.num_steps, args.num_envs))
    rewards = torch.zeros((args.num_steps, args.num_envs))
    dones = torch.zeros((args.num_steps, args.num_envs))
    values = torch.zeros((args.num_steps, args.num_envs))


    global_step = 0

    start_time = time.time()

    # next_obs = torch.Tensor(envs.reset()[0])
    next_obs, info = envs.reset()
    next_obs = torch.Tensor(next_obs)

    action_mask = get_mask(info, args.num_envs, n_actions)
    next_obs = encode_state(next_obs, n_states)
    next_done = torch.zeros(args.num_envs)
    # num_updates = args.total_timesteps // args.batch_size


    global_step = 0
    episode_number = 0
    start_index = 0
    while episode_number < args.max_episodes:
        total_reward = 0
        episode_length = -1
    
        for step in range(0, args.num_steps):
            obs[step] = next_obs
            dones[step] = next_done 
            action_masks[step] = torch.Tensor(action_mask)
        
            with torch.no_grad():
                action, logprob, _,  value = agent.get_action_and_value(next_obs, action_mask = action_mask)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob    
            next_obs, reward, done, term, info = envs.step(action.numpy())
            global_step += 1
            action_mask = get_mask(info, args.num_envs, n_actions)
            rewards[step] = torch.tensor(reward).view(-1)
            next_obs, next_done = torch.Tensor(next_obs), torch.Tensor(done)
            next_obs = encode_state(next_obs, n_states)
            if done:
                # print(start_index, global_step)
                total_reward = info['final_info'][0]['episode']['r']
                episode_length = info['final_info'][0]['episode']['l']
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
                if episode_number >= args.max_episodes:
                    break
         

        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1,-1)
            advantages = torch.zeros_like(rewards)
            lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - next_done   
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t+1]
                    nextvalues = values[t+1]    
                delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + values

        # flatten the batch
        b_obs = obs.reshape((-1,) + (n_states,))
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_action_masks = action_masks.reshape((-1,) + (n_actions,))
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # Optimizing the policy and value network
        b_inds = np.arange(args.batch_size)
        clipfracs = []
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds], b_actions.long()[mb_inds], b_action_masks[mb_inds])
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()


            if args.target_kl is not None:
                if approx_kl > args.target_kl:
                    break

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y
        if use_tensorboard:
        # Logg statistics
            writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
            writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
            writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
            writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
            writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
            writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
            writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
            writer.add_scalar("losses/explained_variance", explained_var, global_step)
            writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)
        print(f"SPS: {int(global_step / (time.time() - start_time))}, Episode: {int(episode_number)}, Step: {global_step}, Return: {total_reward}, Episode length: {episode_length}")

            



    assert (returns_ >= 0.0).all(), "returns should be non-negative"
    assert (discounted_returns_ >= 0.0).all(), "discounted returns should be non-negative"
    envs.close()
    if use_tensorboard:
        writer.close()
    return returns_, discounted_returns_, num_steps
    # writer.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0)  
    parser.add_argument('--num_envs', type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=2.5e-4,
        help="the learning rate of the optimizer")
    # FIXME this is 500 for us I believe check once. 
    parser.add_argument("--num-steps", type=int, default=500,
        help="the number of steps to run in each environment per policy rollout")
    # parser.add_argument("--total-timesteps", type=int, default=5000000,
    #     help="total timesteps of the experiments")
    parser.add_argument("--max-episodes", type=int, default=1000)
    parser.add_argument("--num-minibatches", type=int, default=4,
        help="the number of mini-batches")
    parser.add_argument("--gamma", type=float, default=1.0,
        help="the discount factor gamma")
    parser.add_argument("--gae-lambda", type=float, default=0.95,
        help="the lambda for the general advantage estimation")
    parser.add_argument("--update-epochs", type=int, default=4,
        help="the K epochs to update the policy")
    parser.add_argument("--norm-adv", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggles advantages normalization")
    parser.add_argument("--clip-coef", type=float, default=0.2,
        help="the surrogate clipping coefficient")
    parser.add_argument("--clip-vloss", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggles whether or not to use a clipped loss for the value function, as per the paper.")
    parser.add_argument("--ent-coef", type=float, default=1.0,
        help="coefficient of the entropy")
    parser.add_argument("--vf-coef", type=float, default=0.5,
        help="coefficient of the value function")
    parser.add_argument("--max-grad-norm", type=float, default=0.5,
        help="the maximum norm for the gradient clipping")
    parser.add_argument("--target-kl", type=float, default=None,
        help="the target KL divergence threshold")
    parser.add_argument("--track", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="if toggled, this experiment will be tracked with Weights and Biases")

    args = parser.parse_args()
    assert args.num_envs == 1, "vectorized envs are not supported at the moment" # for purpose of fair evaluation
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)

    run_ppo(args, use_tensorboard=True, use_wandb=args.track)