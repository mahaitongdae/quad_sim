import numpy as np
import torch
import gym
import argparse
import os
import datetime
import yaml
import pickle as pkl

from tensorboardX import SummaryWriter

from train.utils import util, buffer
from train.agent.sac import sac_agent
from train.agent.feature_sac import feature_sac_agent
from environments.quadrotor import QuadrotorEnv

root_dir = os.path.dirname(os.path.abspath(__file__))

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", default='sac_hid_256', type=str)
    parser.add_argument("--alg", default="sac")  # Alg name (sac, feature_sac)
    parser.add_argument("--env", default="Quadrotor-v2")  # Environment name
    parser.add_argument("--env_params_name", default="sac_baseline_randomize_t2w15_35.yml", type=str)
    parser.add_argument("--seed", default=1, type=int)  # Sets Gym, PyTorch and Numpy seeds
    parser.add_argument("--start_timesteps", default=25e3, type=float)  # Time steps initial random policy is used
    parser.add_argument("--eval_freq", default=2e4, type=int)  # How often (time steps) we evaluate
    parser.add_argument("--max_timesteps", default=8e5, type=float)  # Max time steps to run environment
    parser.add_argument("--expl_noise", default=0.1)  # Std of Gaussian exploration noise
    parser.add_argument("--batch_size", default=256, type=int)  # Batch size for both actor and critic
    parser.add_argument("--hidden_dim", default=256, type=int)  # Network hidden dims
    parser.add_argument("--feature_dim", default=1024, type=int)  # Latent feature dim
    parser.add_argument("--discount", default=0.99)  # Discount factor
    parser.add_argument("--tau", default=0.005)  # Target network update rate
    parser.add_argument("--learn_bonus", action="store_true")  # Save model and optimizer parameters
    parser.add_argument("--save_model", action="store_true")  # Save model and optimizer parameters
    parser.add_argument("--extra_feature_steps", default=3, type=int)
    args = parser.parse_args()

    # load env params
    if "Quadrotor" in args.env:
        params_path = root_dir + '/environments/config/' + args.env_params_name
        yaml_stream = open(params_path, 'r')
        params = yaml.load(yaml_stream, Loader=yaml.Loader)
        env = gym.make(args.env, **params['variant']["env_param"])
        from gym.wrappers.transform_reward import TransformReward
        env = TransformReward(env, lambda r: 50. * r )
        params['variant']["env_param"]['init_random_state'] = False
        eval_env = gym.make(args.env, **params['variant']["env_param"])

    else:
        env = gym.make(args.env)
        eval_env = gym.make(args.env)

    
   
    env.seed(args.seed)
    eval_env.seed(args.seed)
    # max_length = env._max_episode_steps

    # setup log
    # dir_name =
    log_path = f'log/{args.env}/{args.alg}/{args.dir}/{args.seed}'
    summary_writer = SummaryWriter(log_path)

    # Store training parameters
    kwargs = vars(args)
    if "Quadrotor" in args.env:
        kwargs.update({'env_params': params['variant']["env_param"]})
    with open(os.path.join(log_path, 'train_params.pkl'), 'wb') as fp:
        pkl.dump(kwargs, fp)

    # set seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    #
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    kwargs = {
        "state_dim": state_dim,
        "action_dim": action_dim,
        "action_space": env.action_space,
        "discount": args.discount,
        "tau": args.tau,
        "hidden_dim": args.hidden_dim,
    }

    # Initialize policy
    if args.alg == "sac":
        agent = sac_agent.SACAgent(**kwargs)
    elif args.alg == 'mle':
        kwargs['extra_feature_steps'] = args.extra_feature_steps
        kwargs['feature_dim'] = args.feature_dim
        agent = feature_sac_agent.MLEFeatureAgent(**kwargs)
    elif args.alg == 'speder':
        kwargs['extra_feature_steps'] = args.extra_feature_steps
        kwargs['feature_dim'] = args.feature_dim
        agent = feature_sac_agent.SPEDERAgent(**kwargs)
    elif args.alg == 'spederv2':
        kwargs['extra_feature_steps'] = args.extra_feature_steps
        kwargs['feature_dim'] = args.feature_dim
        agent = feature_sac_agent.SPEDERAgentV2(**kwargs)
    elif args.alg == 'spederv3':
        kwargs['extra_feature_steps'] = args.extra_feature_steps
        kwargs['feature_dim'] = args.feature_dim
        agent = feature_sac_agent.SPEDERAgentV3(**kwargs)

    replay_buffer = buffer.ReplayBuffer(state_dim, action_dim)

    # Evaluate untrained policy
    evaluations = [util.eval_policy(agent, eval_env)]

    state, done = env.reset(), False
    episode_reward = 0
    episode_timesteps = 0
    episode_num = 0
    timer = util.Timer()

    best_eval_ret = -1e6

    for t in range(int(args.max_timesteps)):

        episode_timesteps += 1

        # Select action randomly or according to policy
        if t < args.start_timesteps:
            action = env.action_space.sample()
        else:
            action = agent.select_action(state, explore=True)

        # Perform action
        next_state, reward, done, _ = env.step(action)
        done_bool = float(done) # if episode_timesteps < max_length else 0

        # Store data in replay buffer
        replay_buffer.add(state, action, next_state, reward, done_bool)

        state = next_state
        episode_reward += reward

        # Train agent after collecting sufficient data
        if t >= args.start_timesteps:
            info = agent.train(replay_buffer, batch_size=args.batch_size)

            if (t+1) % 1000 == 0: # add more frequent logging for train stats.
                for key, value in info.items():
                    summary_writer.add_scalar(f'info/{key}', value, t + 1)
                summary_writer.flush()

        if done:
            # +1 to account for 0 indexing. +0 on ep_timesteps since it will increment +1 even if done=True
            print(
                f"Total T: {t + 1} Episode Num: {episode_num + 1} Episode T: {episode_timesteps} Reward: {episode_reward:.3f}")
            # Reset environment
            state, done = env.reset(), False
            episode_reward = 0
            episode_timesteps = 0
            episode_num += 1

            # Evaluate episode
        if (t + 1) % args.eval_freq == 0:
            steps_per_sec = timer.steps_per_sec(t + 1)
            eva_ret = util.eval_policy(agent, eval_env)
            evaluations.append(eva_ret)

            if t >= args.start_timesteps:
                info['evaluation'] = eva_ret
                for key, value in info.items():
                    summary_writer.add_scalar(f'info/{key}', value, t + 1)
                summary_writer.flush()

            if eva_ret > best_eval_ret:
                best_actor = agent.actor.state_dict()
                best_critic = agent.critic.state_dict()
                torch.save(best_actor, os.path.join(log_path, 'best_actor.pth'))
                torch.save(best_critic, os.path.join(log_path, 'best_critic.pth'))

                if args.alg != 'sac':
                    best_feature_phi = agent.feature_phi.state_dict()
                    best_feature_mu = agent.feature_mu.state_dict()
                    torch.save(best_feature_phi, os.path.join(log_path, 'best_feature_phi.pth'))
                    torch.save(best_feature_mu, os.path.join(log_path, 'best_feature_mu.pth'))
            
            if t >= int(args.max_timesteps) - 5:
                terminal_actor = agent.actor.state_dict()
                terminal_critic = agent.critic.state_dict()
                torch.save(best_actor, os.path.join(log_path, 'terminal_actor_{}.pth'.format(t)))
                torch.save(best_critic, os.path.join(log_path, 'terminal_critic_{}.pth'.format(t)))

                if args.alg != 'sac':
                    best_feature_phi = agent.feature_phi.state_dict()
                    best_feature_mu = agent.feature_mu.state_dict()
                    torch.save(best_feature_phi, os.path.join(log_path, 'terminal_phi_{}.pth'.format(t)))
                    torch.save(best_feature_mu, os.path.join(log_path, 'terminal_mu_{}.pth'.format(t)))

            print('Step {}. Steps per sec: {:.4g}.'.format(t + 1, steps_per_sec))

    summary_writer.close()

    print('Total time cost {:.4g}s.'.format(timer.time_cost()))

    torch.save(agent.actor.state_dict(), os.path.join(log_path, 'last_actor.pth'))
    torch.save(agent.critic.state_dict(), os.path.join(log_path, 'last_critic.pth'))
    if args.alg != 'sac':
        torch.save(agent.feature_phi.state_dict(), os.path.join(log_path, 'last_feature_phi.pth'))
        torch.save(agent.feature_mu.state_dict(), os.path.join(log_path, 'last_feature_mu.pth'))
