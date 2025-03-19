import numpy as np
import math

import torch
from torch import nn
import torch.nn.functional as F
import torch.nn.init as init

from train.utils import util

class phiNet(nn.Module):
  """phi(s,a) network"""
  def __init__(self, obs_dim, action_dim, hidden_dim, output_dim, hidden_depth):
    super().__init__()

    self.nn = util.mlp(obs_dim + action_dim, hidden_dim, output_dim, hidden_depth)
   

    self.outputs = dict()
    self.apply(util.weight_init)

  def forward(self, obs, action):
    assert obs.size(0) == action.size(0)

    # print("obs", obs)
    obs_action = torch.cat([obs, action], dim=-1)
    phi = self.nn(obs_action)

    return phi


class randMu2(nn.Module):
  """mu(s') random function, trying to sample from posterior"""
  def __init__(self, obs_dim, rf_dim, output_dim, sigma = 1.):
    super().__init__()
    fourier_feats = nn.Linear(obs_dim, rf_dim)
    init.normal_(fourier_feats.weight, std = 1./sigma)
    init.uniform_(fourier_feats.bias, 0, 2 * np.pi)
    fourier_feats.weight.requires_grad = False
    fourier_feats.bias.requires_grad = False
    self.fourier = fourier_feats
    print("self.fourier weights", self.fourier.weight)
    rand_weights = nn.Linear(rf_dim, output_dim)
    init.normal_(rand_weights.weight, std = 1.0)
    init.constant_(rand_weights.bias, 0)
    rand_weights.weight.requires_grad = False
    rand_weights.bias.requires_grad = False
    self.rand_weights = rand_weights
    self.rf_dim = rf_dim

  def forward(self, states:torch.Tensor):
    output = math.sqrt(1./self.rf_dim) * self.rand_weights(torch.cos(self.fourier(states)))
    return output