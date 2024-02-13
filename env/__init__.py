from env.quadrotor import QuadrotorEnv
from gym.envs.registration import register

register(id='Quadrotor-v1', 
         entry_point="env:QuadrotorEnv")