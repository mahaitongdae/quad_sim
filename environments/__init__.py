from environments.quadrotor import QuadrotorEnv
from environments.quadrotor_v2 import QuadrotorEnvV2
from gym.envs.registration import register

register(id='Quadrotor-v1', 
         entry_point="environments:QuadrotorEnv")
register(id='Quadrotor-v2', 
         entry_point="environments:QuadrotorEnvV2")