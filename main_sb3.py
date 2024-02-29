import numpy as np
import torch
import gym
import argparse
import os
import datetime
import yaml
import pickle as pkl

from tensorboardX import SummaryWriter

from stable_baselines3 import SAC, PPO
from train.utils import util, buffer
from train.agent.sac import sac_agent
from train.agent.feature_sac import feature_sac_agent
from environments.quadrotor import QuadrotorEnv

root_dir = os.path.dirname(os.path.abspath(__file__))

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", default='sb3_sac', type=str)
    parser.add_argument("--alg", default="sac")  # Alg name (sac, feature_sac)
    parser.add_argument("--env", default="Quadrotor-v2")  # Environment name
    parser.add_argument("--env_params_name", default="ppo_no_damping_add_spin.yml", type=str)
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
    log_path = f'log/sb3/{args.env}'
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

    # model = PPO("MlpPolicy",
    #             env,
    #             verbose=1,
    #             tensorboard_log=log_path,
    #             device='cuda',
    #             batch_size= 256,
    #             learning_rate=3e-4,
    #             )
    # model.load('/home/mht/sim_to_real/training/log/Quadrotor-v2/sac/sb3_sac/1/sb3_ppo.zip')
    # model.learn(total_timesteps=3000000) # Typically not enough
    # model.save(os.path.join(log_path, 'ppo_no_damping_add_spin'))

    model = SAC("MlpPolicy", env,
                verbose=1,
                tensorboard_log=log_path,
                device='cuda',
                batch_size=256,
                learning_rate=3e-4,)

    model.learn(total_timesteps=3000000)  # Typically not enough
    model.save(os.path.join(log_path, 'sac_no_damping_add_spin'))

