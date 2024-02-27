import pickle as pkl
import numpy as np
# from envs.env_helper import *
import argparse
import os
import torch
from train.agent.feature_sac import feature_sac_agent
from train.agent.sac import sac_agent
from train.agent.sac.actor import DiagGaussianActor
from train.utils.util import eval_policy, to_np
import yaml
import gym
root_dir = os.path.dirname(os.path.abspath(__file__))

def eval(log_path, ):
    # with open(os.path.join(log_path, 'train_params.pkl'), 'rb') as f:        
    #     kwargs = pkl.load(f)
    # env_name = kwargs['env']
    
    # if 'env_params' in kwargs.keys():
    #     pass
    # else:
    env_params_name = 'sac_baseline_randomize_t2w15_35.yml'

    # if "Quadrotor" in args.env:
    env = 'Quadrotor-v2'
    params_path = root_dir + '/environments/config/' + env_params_name
    yaml_stream = open(params_path, 'r')
    params = yaml.load(yaml_stream, Loader=yaml.Loader)
    env = gym.make(env, **params['variant']["env_param"])
    from gym.wrappers.transform_reward import TransformReward
    env = TransformReward(env, lambda r: 10. * r)


    actor = DiagGaussianActor(obs_dim=18,
                              action_dim=4,
                              hidden_dim=64,
                              hidden_depth=2,
                              log_std_bounds=[-5., 2.])
    actor.load_state_dict(torch.load(log_path+"/best_actor.pth", map_location=torch.device('cpu')))
    done = False
    state = env.reset()
    env.render()

    def select_action(actor, state):
        state = torch.FloatTensor(state)
        state = state.unsqueeze(0)
        dist = actor(state)
        action = dist.mean
        assert action.ndim == 2 and action.shape[0] == 1
        return to_np(action[0])

    while not done:
        env.step(action=select_action(actor, state))
        env.render()



if __name__ == '__main__':
    eval("log/Quadrotor-v2/sac/sac_with_rescale_init_state_1825/1")