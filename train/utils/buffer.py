import collections
import numpy as np
import torch
import socket

from train.utils import cfusdlog
import matplotlib.pyplot as plt
import re
import argparse
import seaborn as sns
import pandas as pd


Batch = collections.namedtuple(
	'Batch',
	['state', 'action', 'reward', 'next_state', 'done']
	)


class ReplayBuffer(object):
	def __init__(self, state_dim, action_dim, max_size=int(1e6), device='cpu'):
		self.max_size = max_size
		self.ptr = 0
		self.size = 0

		self.state = np.zeros((max_size, state_dim))
		self.action = np.zeros((max_size, action_dim))
		self.next_state = np.zeros((max_size, state_dim))
		self.reward = np.zeros((max_size, 1))
		self.done = np.zeros((max_size, 1))

		device_name = socket.gethostname()
		if device_name.startswith('naliseas'):
			from train import CUDA_DEVICE_WORKSTATION
			self.device = torch.device(CUDA_DEVICE_WORKSTATION if torch.cuda.is_available() else "cpu")
		else:
			self.device = torch.device(device)
		

	def add(self, state, action, next_state, reward, done):
		self.state[self.ptr] = state
		self.action[self.ptr] = action
		self.next_state[self.ptr] = next_state
		self.reward[self.ptr] = reward
		self.done[self.ptr] = done

		self.ptr = (self.ptr + 1) % self.max_size
		self.size = min(self.size + 1, self.max_size)


	def sample(self, batch_size):
		ind = np.random.randint(0, self.size, size=batch_size)

		return Batch(
			state=torch.FloatTensor(self.state[ind]).to(self.device),
			action=torch.FloatTensor(self.action[ind]).to(self.device),
			next_state=torch.FloatTensor(self.next_state[ind]).to(self.device),
			reward=torch.FloatTensor(self.reward[ind]).to(self.device),
			done=torch.FloatTensor(self.done[ind]).to(self.device),
		)

class RealDataBuffer(ReplayBuffer):

	def __init__(self, state_dim, action_dim, max_size=int(1e6)):
		super(RealDataBuffer, self).__init__(state_dim, action_dim, max_size)

	def load_usd_data(self, filename):
		# decode binary log data
		rawData = cfusdlog.decode(filename)
		rawData = rawData['fixedFrequency']
		cmd = rawData['ctrlMel.cmd_thrust']
		start_idx = np.nonzero(cmd)[0][0]
		rawData = rawData[start_idx:]
		xyz = np.hstack([rawData['stateEstimate.x'], rawData['stateEstimate.y'], rawData['stateEstimate.z']]).T
		rpy = np.hstack([rawData['stabilizer.roll'], rawData['stabilizer.pitch'], rawData['stabilizer.yaw']]).T
		vxyz = np.hstack([rawData['stateEstimate.vx'], rawData['stateEstimate.vy'], rawData['stateEstimate.vz']]).T
		rpy_rate = np.hstack([rawData['stateEstimateZ.rateRoll'], rawData['stateEstimateZ.ratePitch'],
							  rawData['stateEstimateZ.rateYaw']]).T / 1000. # rate in milliradians
		cmd_before_mix = np.hstack([rawData['ctrlMel.cmd_roll'],
									rawData['ctrlMel.cmd_pitch'], rawData['ctrl.cmd_yaw']])
		cmd_after_mix = rawData['ctrlMel.cmd_thrust'] + self.MIXER_MATRIX @ cmd_before_mix
		action = cmd_after_mix.T / 65535
		error_pos = xyz - np.array([0.0, 0.0, 1.2])


