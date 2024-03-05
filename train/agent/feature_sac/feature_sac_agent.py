import copy
import torch
from torch import nn
import torch.nn.functional as F
from torch.distributions import Normal
import os

from train.utils.util import unpack_batch  # , # RunningMeanStd
from train.networks.policy import GaussianPolicy
from train.networks.vae import Encoder, Decoder, GaussianFeature
from train.agent.sac.sac_agent import SACAgent
from train.agent.sac.critic import CriticwithPhi
from train.networks.features import MLPFeatureMu, MLPFeaturePhi
from train import CUDA_DEVICE_WORKSTATION
import socket

device_name = socket.gethostname()
if device_name.startswith('naliseas'):
    device = torch.device(CUDA_DEVICE_WORKSTATION if torch.cuda.is_available() else "cpu")
else:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Critic(nn.Module):
    """
	Critic with random fourier features
	"""

    def __init__(
            self,
            feature_dim,
            num_noise=20,
            hidden_dim=256,
    ):
        super().__init__()
        self.num_noise = num_noise
        self.noise = torch.randn(
            [self.num_noise, feature_dim], requires_grad=False, device=device)

        # Q1
        self.l1 = nn.Linear(feature_dim, hidden_dim)  # random feature
        self.l2 = nn.Linear(hidden_dim, hidden_dim)
        self.l3 = nn.Linear(hidden_dim, 1)

        # Q2
        self.l4 = nn.Linear(feature_dim, hidden_dim)  # random feature
        self.l5 = nn.Linear(hidden_dim, hidden_dim)
        self.l6 = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        """
		"""
        # std = log_std.exp()
        # batch_size, d = mean.shape
        #
        # x = mean[:, None, :] + std[:, None, :] * self.noise
        # x = x.reshape(-1, d)

        q1 = F.elu(self.l1(x))  # F.relu(self.l1(x))
        # q1 = q1.reshape([batch_size, self.num_noise, -1]).mean(dim=1)
        q1 = F.elu(self.l2(q1))  # F.relu(self.l2(q1))
        q1 = self.l3(q1)

        q2 = F.elu(self.l4(x))  # F.relu(self.l4(x))
        # q2 = q2.reshape([batch_size, self.num_noise, -1]).mean(dim=1)
        q2 = F.elu(self.l5(q2))  # F.relu(self.l5(q2))
        q2 = self.l3(q2)

        return q1, q2


class LineaCritic(nn.Module):
    """
	Critic with linear
	"""

    def __init__(
            self,
            feature_dim,
    ):
        super().__init__()

        # Q1
        self.l1 = nn.Linear(feature_dim, 1)

        # Q2
        self.l2 = nn.Linear(feature_dim, 1)

    def forward(self, x):
        """
		"""

        q1 = self.l1(x)
        q2 = self.l2(x)

        return q1, q2


class MLEFeatureAgent(SACAgent):
    """
	SAC with VAE learned latent features
	"""

    def __init__(
            self,
            state_dim,
            action_dim,
            action_space,
            lr=1e-4,
            discount=0.99,
            target_update_period=2,
            tau=0.005,
            alpha=0.1,
            auto_entropy_tuning=True,
            hidden_dim=256,
            feature_tau=0.001,
            feature_dim=256,  # latent feature dim
            use_feature_target=True,
            extra_feature_steps=1,
            linear_critic=False
    ):

        super().__init__(
            state_dim=state_dim,
            action_dim=action_dim,
            action_space=action_space,
            lr=lr,
            tau=tau,
            alpha=alpha,
            discount=discount,
            target_update_period=target_update_period,
            auto_entropy_tuning=auto_entropy_tuning,
            hidden_dim=hidden_dim,
        )

        self.feature_dim = feature_dim
        self.feature_tau = feature_tau
        self.use_feature_target = use_feature_target
        self.extra_feature_steps = extra_feature_steps

        self.feature_phi = MLPFeaturePhi(state_dim=state_dim, action_dim=action_dim, hidden_dim=hidden_dim, feature_dim=feature_dim).to(device)
        self.feature_mu = MLPFeatureMu(state_dim=state_dim, hidden_dim=hidden_dim, feature_dim=feature_dim).to(device)

        if use_feature_target:
            self.feature_phi_target = copy.deepcopy(self.feature_phi)
            self.feature_mu_target = self.feature_mu
        self.feature_optimizer = torch.optim.Adam(
            list(self.feature_phi.parameters()) + list(self.feature_mu.parameters()),
            lr=lr,
            weight_decay=1e-2
        )

        if not linear_critic:
            self.critic = Critic(feature_dim=feature_dim, hidden_dim=hidden_dim).to(device)
        else:
            self.critic = LineaCritic(feature_dim=feature_dim).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=lr, betas=[0.9, 0.999])

    def feature_step(self, batch):
        # loss
        phi = self.feature_phi(batch.state, batch.action)
        mu = self.feature_mu(batch.next_state)
        model_learning_loss1 = torch.neg(torch.log(torch.sum(phi * mu, dim=-1)))
        loss = model_learning_loss1.mean()
        phi_norm = torch.norm(phi, dim=-1).mean()

        self.feature_optimizer.zero_grad()
        loss.backward()
        self.feature_optimizer.step()

        return {
            'feature_loss': loss.item(),
            'phi_norm': phi_norm.item()
        }

    def update_actor_and_alpha(self, batch):
        """
		Actor update step
		"""
        dist = self.actor(batch.state)
        action = dist.rsample()
        log_prob = dist.log_prob(action).sum(-1, keepdim=True)

        if self.use_feature_target:
            phi = self.feature_phi_target(batch.state, action)
        else:
            phi = self.feature_phi(batch.state, action)
        q1, q2 = self.critic(phi)
        q = torch.min(q1, q2)

        actor_loss = ((self.alpha) * log_prob - q).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        info = {'actor_loss': actor_loss.item()}

        if self.learnable_temperature:
            self.log_alpha_optimizer.zero_grad()
            alpha_loss = (self.alpha *
                          (-log_prob - self.target_entropy).detach()).mean()
            alpha_loss.backward()
            self.log_alpha_optimizer.step()

            info['alpha_loss'] = alpha_loss
            info['alpha'] = self.alpha

        return info

    def critic_step(self, batch):
        """
		Critic update step
		"""
        state, action, next_state, reward, done = unpack_batch(batch)

        with torch.no_grad():
            dist = self.actor(next_state)
            next_action = dist.rsample()
            next_action_log_pi = dist.log_prob(next_action).sum(-1, keepdim=True)

            if self.use_feature_target:
                phi = self.feature_phi_target(state, action)
                next_phi = self.feature_phi_target(next_state, next_action)
            else:
                phi = self.feature_phi(state, action)
                next_phi = self.feature_phi(next_state, next_action)

            next_q1, next_q2 = self.critic_target(next_phi)
            next_q = torch.min(next_q1, next_q2) - self.alpha * next_action_log_pi
            target_q = reward + (1. - done) * self.discount * next_q

        q1, q2 = self.critic(phi)
        q1_loss = F.mse_loss(target_q, q1)
        q2_loss = F.mse_loss(target_q, q2)
        q_loss = q1_loss + q2_loss

        self.critic_optimizer.zero_grad()
        q_loss.backward()
        self.critic_optimizer.step()

        return {
            'q1_loss': q1_loss.item(),
            'q2_loss': q2_loss.item(),
            'q1': q1.mean().item(),
            'q2': q2.mean().item()
        }

    def update_feature_target(self):
        for param, target_param in zip(self.feature_phi.parameters(), self.feature_phi_target.parameters()):
            target_param.data.copy_(self.feature_tau * param.data + (1 - self.feature_tau) * target_param.data)
        for param, target_param in zip(self.feature_mu.parameters(), self.feature_mu_target.parameters()):
            target_param.data.copy_(self.feature_tau * param.data + (1 - self.feature_tau) * target_param.data)

    def train(self, buffer, batch_size):
        """
		One train step
		"""
        self.steps += 1

        # Feature step
        for _ in range(self.extra_feature_steps + 1):
            batch = buffer.sample(batch_size)
            feature_info = self.feature_step(batch)

            # Update the feature network if needed
            if self.use_feature_target:
                self.update_feature_target()

        # Acritic step
        critic_info = self.critic_step(batch)

        # Actor and alpha step
        actor_info = self.update_actor_and_alpha(batch)

        # Update the frozen target models
        self.update_target()

        return {
            **feature_info,
            **critic_info,
            **actor_info,
        }


class SPEDERAgent(MLEFeatureAgent):

    def feature_step(self, batch):
        """
		Feature learning step

		KL between two gaussian p1 and p2:

		log sigma_2 - log sigma_1 + sigma_1^2 (mu_1 - mu_2)^2 / 2 sigma_2^2 - 0.5
		"""

        # loss
        phi = self.feature_phi(batch.state, batch.action)
        mu = self.feature_mu(batch.next_state)
        model_learning_loss1 = - torch.sum(phi * mu, dim=-1)
        model_learning_loss2 = 1 / (2 * self.feature_dim) * torch.sum(phi * phi, dim=-1)
        model_learning_loss = model_learning_loss1 + model_learning_loss2
        model_learning_loss = model_learning_loss.mean()

        # penalty
        phi_vec = phi[:, :, None]  # shape: [batch, phidim, 1]
        phi_vec_t = phi[:, None, :]  # shape [batch, 1, phi_dim]
        identity = torch.einsum('bij,bjk->bik', phi_vec, phi_vec_t)
        # batch matrix multiplication, more can see https://pytorch.org/docs/stable/generated/torch.einsum.html#torch.einsum
        identity = torch.mean(identity, dim=0)
        penalty_factor = torch.tensor(1e6, device=device)  # TODO: tune it maybe
        penalty_loss = penalty_factor * F.mse_loss(identity, torch.eye(self.feature_dim).to(device) / self.feature_dim)

        loss = model_learning_loss + penalty_loss

        self.feature_optimizer.zero_grad()
        loss.backward()
        self.feature_optimizer.step()

        return {
            'feature_loss': loss.item(),
            'model_learning_loss1': model_learning_loss1.mean().item(),
            'model_learning_loss2': model_learning_loss2.mean().item(),
            'model_learning_loss': model_learning_loss.item(),
            'penalty_loss': penalty_loss.item(),
            # 's_loss': s_loss.mean().item(),
            # 'r_loss': r_loss.mean().item()
        }

class SPEDERAgentV2(MLEFeatureAgent):
    """
    V2 follows eq (9) in Ren. 2023
    """

    def feature_step(self, batch):
        """
		Feature learning step

		KL between two gaussian p1 and p2:

		log sigma_2 - log sigma_1 + sigma_1^2 (mu_1 - mu_2)^2 / 2 sigma_2^2 - 0.5
		"""

        # loss
        phi = self.feature_phi(batch.state, batch.action)
        mu = self.feature_mu(batch.next_state)
        model_learning_loss1 = - 2. * torch.sum(phi * mu, dim=-1)
        model_learning_loss2 = torch.mean(torch.matmul(phi, mu.T)  ** 2, dim=1)
        model_learning_loss = model_learning_loss1 + model_learning_loss2
        model_learning_loss = model_learning_loss.mean()

        # loss = model_learning_loss

        self.feature_optimizer.zero_grad()
        model_learning_loss.backward()
        self.feature_optimizer.step()

        return {
            'feature_loss': model_learning_loss.item(),
            'model_learning_loss1': model_learning_loss1.mean().item(),
            'model_learning_loss2': model_learning_loss2.mean().item(),
            # 'model_learning_loss': model_learning_loss.item(),
            # 's_loss': s_loss.mean().item(),
            # 'r_loss': r_loss.mean().item()
        }

class SPEDERAgentV3(SACAgent):
    """
	SAC with VAE learned latent features
	"""

    def __init__(
            self,
            state_dim,
            action_dim,
            action_space,
            lr=1e-4,
            discount=0.99,
            target_update_period=2,
            tau=0.005,
            alpha=0.1,
            auto_entropy_tuning=True,
            hidden_dim=256,
            feature_tau=0.001,
            feature_dim=256,  # latent feature dim
            use_feature_target=True,
            extra_feature_steps=1,
            linear_critic=False
    ):

        super().__init__(
            state_dim=state_dim,
            action_dim=action_dim,
            action_space=action_space,
            lr=lr,
            tau=tau,
            alpha=alpha,
            discount=discount,
            target_update_period=target_update_period,
            auto_entropy_tuning=auto_entropy_tuning,
            hidden_dim=hidden_dim,
        )

        self.feature_dim = feature_dim
        self.feature_tau = feature_tau
        self.use_feature_target = use_feature_target
        self.extra_feature_steps = extra_feature_steps

        # self.feature_phi = MLPFeaturePhi(state_dim=state_dim, action_dim=action_dim, hidden_dim=hidden_dim, feature_dim=feature_dim).to(device)
        self.feature_mu = MLPFeatureMu(state_dim=state_dim, hidden_dim=hidden_dim, feature_dim=feature_dim).to(device)

        if use_feature_target:
            # self.feature_phi_target = copy.deepcopy(self.feature_phi)
            self.feature_mu_target = self.feature_mu
        self.feature_optimizer = torch.optim.Adam(
            list(self.feature_mu.parameters()),
            lr=lr,
            weight_decay=1e-2
        )

        self.critic = CriticwithPhi(input_dim=state_dim + action_dim, feature_dim=feature_dim,hidden_dim=hidden_dim,).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=lr, betas=[0.9, 0.999])

    def feature_step(self, batch):
        # loss
        phi, _ = self.critic.get_feature(batch.state, batch.action)
        mu = self.feature_mu(batch.next_state)
        model_learning_loss1 = - 2. * torch.sum(phi * mu, dim=-1)
        model_learning_loss2 = torch.mean(torch.matmul(phi, mu.T) ** 2, dim=1)
        model_learning_loss = model_learning_loss1 + model_learning_loss2
        model_learning_loss = model_learning_loss.mean()

        # loss = model_learning_loss

        self.feature_optimizer.zero_grad()
        model_learning_loss.backward()
        self.feature_optimizer.step()

        return {
            'feature_loss': model_learning_loss.item(),
            'model_learning_loss1': model_learning_loss1.mean().item(),
            'model_learning_loss2': model_learning_loss2.mean().item(),
        }

    def update_feature_target(self):
        for param, target_param in zip(self.feature_mu.parameters(), self.feature_mu_target.parameters()):
            target_param.data.copy_(self.feature_tau * param.data + (1 - self.feature_tau) * target_param.data)

    def train(self, buffer, batch_size):
        """
		One train step
		"""
        self.steps += 1
        batch = buffer.sample(batch_size)

        # Feature step
        # for _ in range(self.extra_feature_steps + 1):
        #     batch = buffer.sample(batch_size)
        #     feature_info = self.feature_step(batch)

        #     # Update the feature network if needed
        #     if self.use_feature_target:
        #         self.update_feature_target()

        # Acritic step
        critic_info = self.critic_step(batch)

        # Actor and alpha step
        actor_info = self.update_actor_and_alpha(batch)

        # Update the frozen target models
        self.update_target()

        return {
            # **feature_info,
            **critic_info,
            **actor_info,
        }

class TransferAgent(SPEDERAgent):

    def __init__(self,
                 log_path,
                 state_dim,
                 action_dim,
                 action_space,
                 lr,
                 linear_critic = False,
                 aug_feature_dim = 128):
        super(TransferAgent, self).__init__(
            state_dim,
            action_dim,
            action_space,
            lr=lr,
            discount=0.99,
            target_update_period=2,
            tau=0.005,
            alpha=0.1,
            auto_entropy_tuning=True,
            hidden_dim=256,
            feature_tau=0.001,
            feature_dim=256,  # latent feature dim
            use_feature_target=True,
            extra_feature_steps=1,
            linear_critic=linear_critic)
        # load nets trained in simulators
        self.feature_phi.load_state_dict(
            torch.load(os.path.join(log_path, 'best_feature_phi.pth'), map_location={'cuda:1': 'cuda:0'}))
        # map location is for trained on workstations and load on locals.
        self.feature_mu.load_state_dict(
            torch.load(os.path.join(log_path, 'best_feature_mu.pth'), map_location={'cuda:1': 'cuda:0'}))
        self.critic.load_state_dict(
            torch.load(os.path.join(log_path, 'best_critic.pth'), map_location={'cuda:1': 'cuda:0'}))
        self.actor.load_state_dict(
            torch.load(os.path.join(log_path, 'best_actor.pth'), map_location={'cuda:1': 'cuda:0'}))

        self.augmented_feature_phi = MLPFeaturePhi(state_dim, action_dim, feature_dim=128)
        self.augmented_feature_mu = MLPFeatureMu(state_dim, action_dim, feature_dim=128)

        if self.use_feature_target:
            self.augmented_feature_phi_target = copy.deepcopy(self.augmented_feature_phi)
            self.augmented_feature_mu_target = copy.deepcopy(self.augmented_feature_mu)

        self.feature_optimizer = torch.optim.Adam(list(self.augmented_feature_phi.parameters())
                                                  + list(self.augmented_feature_mu.parameters()),
            lr=lr,
            weight_decay=1e-2)

        if linear_critic:
            self.augemented_critic = LineaCritic(feature_dim=aug_feature_dim)
        else:
            self.augemented_critic = Critic(feature_dim=aug_feature_dim)
        self.augemented_critic_target = copy.deepcopy(self.augemented_critic)


        self.aug_critic_optimizer = torch.optim.Adam(self.augemented_critic.parameters(), lr=lr, betas=[0.9, 0.999])


    def feature_step(self, batch):
        phi = self.feature_phi(batch.state, batch.action)
        mu = self.feature_mu(batch.next_state)
        model_learning_loss1 = - torch.sum(phi * mu, dim=-1)
        model_learning_loss2 = 1 / (2 * self.feature_dim) * torch.sum(phi * phi, dim=-1)
        model_learning_loss = model_learning_loss1 + model_learning_loss2
        model_learning_loss = model_learning_loss.mean()

        # penalty
        phi_vec = phi[:, :, None]  # shape: [batch, phidim, 1]
        phi_vec_t = phi[:, None, :]  # shape [batch, 1, phi_dim]
        identity = torch.einsum('bij,bjk->bik', phi_vec, phi_vec_t)
        # batch matrix multiplication, more can see https://pytorch.org/docs/stable/generated/torch.einsum.html#torch.einsum
        identity = torch.mean(identity, dim=0)
        penalty_factor = torch.tensor(1e10, device=device)  # TODO: tune it maybe
        penalty_loss = penalty_factor * F.mse_loss(identity, torch.eye(self.feature_dim).to(device) / self.feature_dim)

        loss = model_learning_loss + penalty_loss

        self.feature_optimizer.zero_grad()
        loss.backward()
        self.feature_optimizer.step()

        return {
            'feature_loss': loss.item(),
            'model_learning_loss1': model_learning_loss1.mean().item(),
            'model_learning_loss2': model_learning_loss2.mean().item(),
            'model_learning_loss': model_learning_loss.item(),
            'penalty_loss': penalty_loss.item(),
            # 's_loss': s_loss.mean().item(),
            # 'r_loss': r_loss.mean().item()
        }

