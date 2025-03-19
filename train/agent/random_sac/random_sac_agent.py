import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import copy

from train.utils.util import unpack_batch
from train.utils import util
from train.agent.sac.actor import DiagGaussianActor
from train.agent.random_sac import learn_phi

#create a critic network with frozen phi and trainable weights.
class phiCritic_last_layer(nn.Module):
	"""A trainable last layer"""
	def __init__(self, phi_last_dim = 512):
		super().__init__()
		
		self.norm = nn.LayerNorm(phi_last_dim)
		self.norm.bias.requires_grad = False

		# The trainable last layer (only weights, no bias)
		self.trainable_layer = nn.Linear(phi_last_dim, 1, bias=False)


		
		# # You can initialize the weights if necessary
		# nn.init.xavier_uniform_(self.trainable_layer.weight)

	def forward(self, phi):
		# Get the output from the fixed phiNet
		x = self.norm(phi)
		
		# Apply the trainable last layer
		output = self.trainable_layer(x)
		
		return output, output #for simplicity of testing, just return Q1==Q2
  
def reward_fn(state, action):
    rew_pos = - 2.5 * torch.norm(state[:, 0:3], dim=1)
    rew_rpy = - 0.1 * torch.norm(state[:, 7:10], dim=1)
    rew_lin_vel = - 0.05 * torch.norm(state[:, 10:13], dim=1)
    rew_ang_vel = - 0.05 * torch.norm(state[:, 13:16], dim=1)
    rew_action = - 0.1 * torch.norm(action - 0.1818642897259981, dim=1) # hard_coded for now
    last_action = state[:, -4:]
    rew_action_diff = -0.1 * torch.norm((action - last_action), dim=1)
    
    return torch.exp(rew_pos +
                rew_rpy +
                rew_lin_vel +
                rew_ang_vel +
                rew_action +
                rew_action_diff
                )


class randSACAgent(object):
	"""
	Randomized SVD SAC agent
	"""
	def __init__(
			self, 
			state_dim, 
			action_dim, 
			action_range,
			lr=3e-4,
			discount=0.99, 
			target_update_period=2,
			tau=0.005,
			alpha=0.1,
			auto_entropy_tuning=True,
			hidden_dim=512,
			device='cpu',
			sigma = None,
			has_feature_step = True,
			use_rand_nn_mu = False,
			**kwargs
			):

		self.steps = 0

		self.device = torch.device(device)
		self.action_range = action_range
		self.discount = discount 
		self.tau = tau 
		self.target_update_period = target_update_period
		self.learnable_temperature = auto_entropy_tuning
		self.rsvd_num = kwargs['rsvd_num']
		print("self.rsvd_num", self.rsvd_num)
		self.rf_num = kwargs['rf_num']
		print("self.rf_num", self.rf_num)

		self.has_feature_step = has_feature_step

		# # functions
		# print("This is use_V_critic", use_V_critic)
		# if use_V_critic == False:
		# 	self.critic = phiCritic(critic_phi, self.rsvd_num).to(self.device)
		# else:
		# 	self.critic = phiCritic_V(critic_phi, self.rsvd_num).to(self.device)
		# 	print("I am using RFVCritic")
		# 	self.critic = RFVCritic(s_dim = state_dim, rf_num = self.rsvd_num,sigma = 1.0).to(self.device)
		# # 	self.critic = DoubleVCritic(
		# #     obs_dim=state_dim, 
		# #     hidden_dim = 256,
		# #     hidden_depth=2,
		# #     use_ortho_init=True
		# # ).to(self.device)

		# for name, param in self.critic.named_parameters():
		# 	if param.requires_grad == True:
		# 		print("param requires grad: ", name)
		sigma = 1.0
		print("rsvd mu sigma", sigma)
		self.phi_net = learn_phi.phiNet(state_dim, action_dim, hidden_dim = hidden_dim, output_dim = self.rsvd_num , hidden_depth  = 2).to(self.device)
		if use_rand_nn_mu == True:
			self.rand_mu_net = learn_phi.nnMu(state_dim, output_dim=self.rsvd_num, hidden_dim=256, hidden_depth=1).to(self.device)
		else:
			# self.rand_mu_net = learn_phi.randMu(state_dim, self.rsvd_num, sigma = sigma).to(self.device)
			self.rand_mu_net = learn_phi.randMu2(state_dim, self.rf_num, self.rsvd_num, sigma = sigma).to(self.device)
			# self.rand_mu_net = learn_phi.randMu(state_dim, self.rsvd_num, sigma = 0.3).to(self.device)
		self.phi_optimizer = torch.optim.Adam(self.phi_net.parameters(),
													# lr= 1e-4,
													lr =  3e-4,
													betas=[0.9, 0.999])
		self.critic_last_layer = phiCritic_last_layer(self.rsvd_num).to(self.device)
		critic_lr = kwargs['critic_lr'] if 'critic_lr' in kwargs.keys() else lr
		self.critic_last_layer_optimizer = torch.optim.Adam(self.critic_last_layer.parameters(),
												 lr=critic_lr,
												 betas=[0.9, 0.999])

		self.critic_target = copy.deepcopy([self.phi_net, self.critic_last_layer])
		self.actor = DiagGaussianActor(
			obs_dim=state_dim, 
			action_dim=action_dim,
			hidden_dim=hidden_dim,
			hidden_depth=2,
			log_std_bounds=[-5., 2.],
		).to(self.device)
		self.log_alpha = torch.tensor(np.log(alpha)).float().to(self.device)
		self.log_alpha.requires_grad = True
		self.target_entropy = -action_dim
		
		 # optimizers
		self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),
												lr=lr,
												betas=[0.9, 0.999])

		# self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),
		# 										 lr=critic_lr,
		# 										 betas=[0.9, 0.999])

		self.log_alpha_optimizer = torch.optim.Adam([self.log_alpha],
													lr=lr,
													betas=[0.9, 0.999])


		# self.phi_net = learn_phi.phiNet(state_dim, action_dim, hidden_dim = hidden_dim, output_dim = self.rsvd_num , hidden_depth  = 2).to(self.device)
		# self.rand_mu_net = learn_phi.randMu(state_dim, self.rsvd_num, sigma = 1.0).to(self.device)
		# self.phi_optimizer = torch.optim.Adam(self.phi_net.parameters(),
		# 											lr= 1e-4,
		# 											betas=[0.9, 0.999])
		


	@property
	def alpha(self):
		return self.log_alpha.exp()


	def get_reward(self, state, action):
		reward = reward_fn(state, action)
		return torch.reshape(reward, (reward.shape[0], 1))

	def select_action(self, state, explore=False):
		if isinstance(state, list):
			state = np.array(state)
		state = state.astype(np.float32)
		assert len(state.shape) == 1
		state = torch.from_numpy(state).to(self.device)
		state = state.unsqueeze(0)
		dist = self.actor(state)
		action = dist.sample() if explore else dist.mean
		action = action.clamp(torch.tensor(-1, device=self.device),
							  torch.tensor(1, device=self.device))
		assert action.ndim == 2 and action.shape[0] == 1
		return util.to_np(action[0])

	def batch_select_action(self, state, explore=False):
		assert isinstance(state, torch.Tensor)
		dist = self.actor(state)
		action = dist.sample() if explore else dist.mean
		action = action.clamp(torch.tensor(-1, device=self.device),
							  torch.tensor(1, device=self.device))
		return action


	def update_target(self):
		if self.steps % self.target_update_period == 0:
			for param, target_param in zip(self.phi_net.parameters(), self.critic_target[0].parameters()):
				target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
			for param, target_param in zip(self.critic_last_layer.parameters(), self.critic_target[1].parameters()):
				target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)				




	def critic_step(self, batch):
		"""
		Critic update step
		"""
		# state, action, reward, next_state, next_action, next_reward,next_next_state, done = unpack_batch(batch)
		state, action, next_state, reward, done = unpack_batch(batch)
	
		# for name, param in self.critic.named_parameters():
		# 	print("self critic params norm (%s)" % name, param.norm())

		with torch.no_grad():
			dist = self.actor(next_state)
			next_action = dist.rsample()
			next_action_log_pi = dist.log_prob(next_action).sum(-1, keepdim=True)
			# next_q1, next_q2 = self.critic_target(next_state, next_action)
			# next_q1, next_q2 = self.critic_target(self.dynamics_fn(next_state, next_action))
			next_q1, next_q2 = self.critic_target[1](self.critic_target[0](next_state, next_action))
			next_q = torch.min(next_q1, next_q2) - self.alpha * next_action_log_pi
			next_reward = self.get_reward(next_state, next_action)  # reward for new s,a
			target_q = next_reward + (1. - done) * self.discount * next_q

		q1, q2 = self.critic_last_layer(self.phi_net(state, action))
		q1_loss = F.mse_loss(target_q, q1)
		q2_loss = F.mse_loss(target_q, q2)
		q_loss = q1_loss + q2_loss

		self.critic_last_layer_optimizer.zero_grad()
		# self.phi_optimizer.zero_grad()
		q_loss.backward()
		# for name, param in self.critic_last_layer.named_parameters():
		# 	print("self critic last layer params grad (%s)" % name, param.grad)
		# for name, param in self.phi_net.named_parameters():
		# 	print("self phi net params grad (%s)" % name, param.grad)
		self.critic_last_layer_optimizer.step()
		# self.phi_optimizer.step()

		info = {
			'q1_loss': q1_loss.item(),
			'q2_loss': q2_loss.item(),
			'q1': q1.mean().item(),
			'q2': q2.mean().item(),
			'layer_norm_weights_norm': self.critic_last_layer.norm.weight.norm(),
		}

		dist = {
			'td_error': (torch.min(q1, q2) - target_q).cpu().detach().clone().numpy(),
			'q': torch.min(q1, q2).cpu().detach().clone().numpy()
		}

		info.update({'critic_dist': dist})

		return info


	def update_actor_and_alpha(self, batch):
		"""
		Actor update step
		"""
		# dist = self.actor(batch.state, batch.next_state)
		dist = self.actor(batch.state)
		action = dist.rsample()
		log_prob = dist.log_prob(action).sum(-1, keepdim=True)
		reward = self.get_reward(batch.state, action)  # use reward in q-fn
		# q1, q2 = self.critic(batch.state, action)
		q1, q2 = self.critic_last_layer(self.phi_net(batch.state, action))
		q = self.discount * torch.min(q1, q2) + reward

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


	def feature_step(self, batch):
		"""
		Feature update step
		"""
		# state, action, reward, next_state, next_action, next_reward,next_next_state, done = unpack_batch(batch)
		state, action, next_state, reward, done = unpack_batch(batch)
	

		phi = self.phi_net(state,action)
		with torch.no_grad():
			rand_mu = self.rand_mu_net(next_state)
		# print("phi shape", phi.shape)
		inner_prod = torch.sum(phi * rand_mu, dim = 1)
		loss_norm = torch.mean(torch.sum(phi**2,dim=1))
		# print("loss_norm", loss_norm)
		# print("inner_prod shape", inner_prod.shape)
		loss_self = -2 * torch.mean(inner_prod)
		# print("loss_self", loss_self)
		loss = loss_norm + loss_self
		# loss = loss_norm
		self.phi_optimizer.zero_grad()
		loss.backward()
		self.phi_optimizer.step()
		info = {
			'loss_norm': loss_norm.item(),
			'loss_self': loss_self.item()
		}
		return info




	def train(self, buffer, batch_size):
		"""
		One train step
		"""
		self.steps += 1

		batch = buffer.sample(batch_size)

		# Feature info
		if self.has_feature_step == True:
			feature_info = self.feature_step(batch)

		# Acritic step
		critic_info = self.critic_step(batch)

		# Actor and alpha step
		actor_info = self.update_actor_and_alpha(batch)


		# Update the frozen target models
		self.update_target()

		if self.has_feature_step == True:
			return {
				**feature_info,
				**critic_info, 
				**actor_info,
			}
		else:
			return {
				**critic_info, 
				**actor_info,
			}

	def batch_train(self, batch):

		"""
				One train step
				"""
		self.steps += 1

		#feature step
		# feature_info = self.feature_step(batch)

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