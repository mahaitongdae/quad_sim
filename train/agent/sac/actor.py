"""
We adapt the code from https://github.com/denisyarats/pytorch_sac
"""

import numpy as np
import torch
import math
from torch import nn
import torch.nn.functional as F
from torch import distributions as pyd

from train.utils import util


class TanhTransform(pyd.transforms.Transform):
    domain = pyd.constraints.real
    codomain = pyd.constraints.interval(-1.0, 1.0)
    bijective = True
    sign = +1

    def __init__(self, cache_size=1):
        super().__init__(cache_size=cache_size)

    @staticmethod
    def atanh(x):
        return 0.5 * (x.log1p() - (-x).log1p())

    def __eq__(self, other):
        return isinstance(other, TanhTransform)

    def _call(self, x):
        return x.tanh()

    def _inverse(self, y):
        # We do not clamp to the boundary here as it may degrade the performance of certain algorithms.
        # one should use `cache_size=1` instead
        return self.atanh(y)

    def log_abs_det_jacobian(self, x, y):
        # We use a formula that is more numerically stable, see details in the following link
        # https://github.com/tensorflow/probability/commit/ef6bb176e0ebd1cf6e25c6b5cecdd2428c22963f#diff-e120f70e92e6741bca649f04fcd907b7
        return 2. * (math.log(2.) - x - F.softplus(-2. * x))


class SquashedNormal(pyd.transformed_distribution.TransformedDistribution):
    def __init__(self, loc, scale):
        self.loc = loc
        self.scale = scale

        self.base_dist = pyd.Normal(loc, scale)
        transforms = [TanhTransform()]
        super().__init__(self.base_dist, transforms)

    @property
    def mean(self):
        mu = self.loc
        for tr in self.transforms:
            mu = tr(mu)
        return mu


class DiagGaussianActor(nn.Module):
    """torch.distributions implementation of an diagonal Gaussian policy."""

    def __init__(self, obs_dim, action_dim, hidden_dim, hidden_depth,
                 log_std_bounds, hidden_activation=nn.ELU(inplace=True), action_range=np.array([-1., 1.]),
                 lipsnet=False):
        super().__init__()
        self.action_range = action_range
        self.log_std_bounds = log_std_bounds
        if not lipsnet:
            self.trunk = util.mlp(obs_dim, hidden_dim, 2 * action_dim,
                                  hidden_depth, hidden_activation=hidden_activation)
        else:
            from train.utils.lipsnet import LipsNet
            self.trunk = LipsNet(f_sizes=[obs_dim, 64, 64, 2 * action_dim], f_hid_nonliear=nn.ReLU,
                                 f_out_nonliear=nn.Identity,
                                 global_lips=False, k_init=1, k_sizes=[obs_dim, 32, 1], k_hid_act=nn.Tanh,
                                 k_out_act=nn.Softplus,
                                 loss_lambda=0.1, eps=1e-4, squash_action=True)

        self.outputs = dict()
        self.apply(util.weight_init)

    def forward(self, obs):
        mu, log_std = self.trunk(obs).chunk(2, dim=-1)

        # constrain log_std inside [log_std_min, log_std_max]
        log_std = torch.tanh(log_std)
        log_std_min, log_std_max = self.log_std_bounds
        log_std = log_std_min + 0.5 * (log_std_max - log_std_min) * (log_std +
                                                                     1)

        std = log_std.exp()

        self.outputs['mu'] = mu
        self.outputs['std'] = std

        dist = SquashedNormal(mu, std)
        return dist


class DifferentiableMellinger(nn.Module):
    cf_mass = 0.027
    massThrust = 132000
    INT16_MAX = 65536

    def __init__(self, ):
        super().__init__(self)
        self.integral_error = torch.zeros([3, ])
        # XY positions
        self.kp_xy = torch.nn.Parameter(torch.tensor(0.4, requires_grad=True))
        self.ki_xy = torch.nn.Parameter(torch.tensor(0.2, requires_grad=True))
        self.kd_xy = torch.nn.Parameter(torch.tensor(0.05, requires_grad=True))
        # Z position
        self.kp_z = torch.nn.Parameter(torch.tensor(1.25, requires_grad=True))
        self.ki_z = torch.nn.Parameter(torch.tensor(0.4, requires_grad=True))
        self.kd_z = torch.nn.Parameter(torch.tensor(0.05, requires_grad=True))
        # Attitude
        self.kR_xy = torch.nn.Parameter(torch.tensor(70000., requires_grad=True))
        self.kw_xy = torch.nn.Parameter(torch.tensor(20000., requires_grad=True))
        self.ki_m_xy = torch.nn.Parameter(torch.tensor(0., requires_grad=True))

        self.kR_z = torch.nn.Parameter(torch.tensor(60000., requires_grad=True))
        self.kw_z = torch.nn.Parameter(torch.tensor(12000., requires_grad=True))
        self.ki_m_z = torch.nn.Parameter(torch.tensor(500., requires_grad=True))

        self.i_range_xy = 2.0
        self.i_range_z = 0.4
        self.i_range_m_xy = 1.0
        self.i_range_m_z = 1500.

        self.i_error_x = torch.tensor(0.)
        self.i_error_y = torch.tensor(0.)
        self.i_error_z = torch.tensor(0.)
        self.i_error_m_x = torch.tensor(0.)
        self.i_error_m_y = torch.tensor(0.)
        self.i_error_m_z = torch.tensor(0.)

    def reset(self):
        self.i_error_x = torch.tensor(0.)
        self.i_error_y = torch.tensor(0.)
        self.i_error_z = torch.tensor(0.)
        self.i_error_m_x = torch.tensor(0.)
        self.i_error_m_y = torch.tensor(0.)
        self.i_error_m_z = torch.tensor(0.)



if __name__ == '__main__':
    actor = DiagGaussianActor(4, 2, 64, 2, np.array([-20, 1]))
    input = 100 * torch.rand(128, 4)
    print(actor(input).scale)
