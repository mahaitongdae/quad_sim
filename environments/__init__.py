from environments.quadrotor import QuadrotorEnv
from gym.envs.registration import register

register(id='Quadrotor-v1', 
         entry_point="environments:QuadrotorEnv")