"""Script demonstrating the use of `gym_pybullet_drones`' Gymnasium interface.

Class HoverAviary is used as a learning env for the PPO algorithm.

Example
-------
In a terminal, run as:

    $ python learn.py

Notes
-----
This is a minimal working example integrating `gym-pybullet-drones` with
reinforcement learning libraries `stable-baselines3`.

"""
import time
import argparse
import gymnasium as gym
import numpy as np
# from stable_baselines3 import PPO, SAC, TD3, A2C

from gym_pybullet_drones.utils.Logger import Logger, LoggerV1
from gym_pybullet_drones.envs.single_agent_rl.HoverAviary import HoverAviary
from gym_pybullet_drones.envs.single_agent_rl.HoverAviaryDelay import HoverAviaryDelay
from gym_pybullet_drones.envs.single_agent_rl.BaseSingleAgentAviary import ActionType
from gym_pybullet_drones.utils.utils import sync, str2bool
from train.agent.sac.actor import DiagGaussianActor
import torch
# from sbx import TQC, SAC
import seaborn
import pandas as pd
from matplotlib import pyplot as plt

DEFAULT_GUI = True
DEFAULT_RECORD_VIDEO = False
DEFAULT_OUTPUT_FOLDER = 'results'
DEFAULT_COLAB = False

def run(output_folder=DEFAULT_OUTPUT_FOLDER, gui=DEFAULT_GUI, plot=True, colab=DEFAULT_COLAB, record_video=DEFAULT_RECORD_VIDEO):


    rew = {'t':[], 'reward': [], 'label':[]}
    #### Show (and record a video of) the model's performance ##
    env = HoverAviary(gui=gui,
                      record=record_video,
                      add_action_obs=True,
                      act=ActionType.PWM,
                      initial_rpys=np.zeros((1,3))
                     )
    env.EPISODE_LEN_SEC = 3
    logger = LoggerV1(logging_freq_hz=int(env.CTRL_FREQ),
                      env = env,
                    num_drones=1,
                    output_folder=output_folder,
                    colab=colab
                    )
    obs, info = env.reset(seed=42, options={})
    start = time.time()
    for i in range(env.CTRL_FREQ):
        # ref https://www.bitcraze.io/images/getting-started/cf2_props.png
        # ref https://www.bitcraze.io/documentation/system/platform/cf2-coordinate-system/
        action = 0.2 * np.ones((4, ))
        # action = np.array([0.1, 0.1, 0.0, 0.0]) # negative roll, checked
        # action = np.array([0.1, 0.0, 0.0, 0.1]) # positive pitch, checked
        # action = np.array([-0.1, 0.1, -0.1, 0.1]) # positive yaw, checked
        obs, reward, terminated, truncated, info = env.step(action)
        logger.log(drone=0,
                   timestamp=i/env.CTRL_FREQ,
                   state=obs,
                   control=action
                   )
        env.render()
        time.sleep(0.02)
        for key, value in info.items():
            rew['reward'].append(value)
            rew['label'].append(key)
            rew['t'].append(i)

        print(terminated)
        sync(i, start, env.CTRL_TIMESTEP)
        if terminated:
            print(info)
            obs, _ = env.reset(seed=42, options={})
    env.close()

    if plot:
        logger.plot()

    df = pd.DataFrame.from_dict(rew)
    seaborn.lineplot(df, x='t', y='reward', hue='label')
    plt.show()

run()