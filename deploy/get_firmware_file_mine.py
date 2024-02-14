import torch
from train.agent.sac.actor import DiagGaussianActor


def get_firmware_file(log_path):

    actor = DiagGaussianActor(obs_dim=18,
                              action_dim=8,
                              hidden_dim=64,
                              hidden_depth=2,
                              log_std_bounds=[-5., 2.]) # hard coded for drone controllers.
    # if use_nystrom == False:
    #     critic = RFVCritic(**kwargs)
    # else:
    #     critic = nystromVCritic(**kwargs)

    actor.load_state_dict(torch.load(log_path+"/actor.pth"))