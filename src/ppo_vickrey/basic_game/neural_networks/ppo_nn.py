from __future__ import print_function
import os
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
import sys
import time
import numpy as np
import copy
import itertools

from random import shuffle
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributions as distributions
from scipy.special import beta, digamma
import torch.nn.utils as utils

sys.path.append('../../')
from Arena import Arena
from utils import *
from tqdm import tqdm
import pdb




# Assume you have a dotdict utility (or substitute with a simple dict)
'''
class dotdict(dict):
	__getattr__ = dict.get
	__setattr__ = dict.__setitem__
'''

# -----------------------------------------------------------------------------
# PPO Network (Policy & Value) for the Auction Game
# -----------------------------------------------------------------------------


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Beta

'''
class AttentionPoolBranch(nn.Module):
	def __init__(self, seq_length, embed_dim=8, num_heads=2):
		super(AttentionPoolBranch, self).__init__()
		self.seq_length = seq_length
		self.embed_dim = embed_dim
		self.attention = nn.MultiheadAttention(
			embed_dim=self.embed_dim, 
			num_heads=num_heads, 
			batch_first=True
		)
		self.fc = nn.Linear(self.embed_dim, 8)

	def forward(self, x):
		# x shape: (B, seq_length, embed_dim)
		B, S, E = x.size()
		attn_output, _ = self.attention(x, x, x)
		return self.fc(attn_output.mean(dim=1))

class PPOAuctionEnvNet(nn.Module):
	def __init__(self, game, pool_seq_length_forward=None, pool_seq_length_defense=None, pool_seq_length_goalie=None):
		super(PPOAuctionEnvNet, self).__init__()
		structure = game.getStructure()
		self.num_players = structure["NUM_PLAYERS"]
		self.game_budget = structure["GAME_BUDGET"]
		self.num_forwards = structure["FORWARDS_NEEDED"]
		self.num_defensemen = structure["DEFENSEMEN_NEEDED"]
		self.num_goalies = structure["GOALIES_NEEDED"]

		self.global_branch = nn.Sequential(
			nn.Linear(7, 16),  # Corrected input dimension
			nn.BatchNorm1d(16),
			nn.SiLU(),
			nn.Linear(16, 16),
			nn.BatchNorm1d(16),
			nn.SiLU()
		)

		# Our Branch
		self.our_branch = nn.Sequential(
			nn.Linear(5, 16),
			nn.BatchNorm1d(16),
			nn.SiLU(),
			nn.Linear(16, 8),
			nn.BatchNorm1d(8),
			nn.SiLU()
		)

		# Opponents Branch
		self.opp_shared = nn.Sequential(
			nn.Linear(5, 16),
			nn.BatchNorm1d(16),
			nn.SiLU(),
			nn.Linear(16, 16),
			nn.BatchNorm1d(16),
			nn.SiLU()
		)
		self.opponents_branch = nn.Sequential(
			nn.Linear((self.num_players-1)*16, 16),
			nn.BatchNorm1d(16),
			nn.SiLU(),
			nn.Linear(16, 8),
			nn.BatchNorm1d(8),
			nn.SiLU()
		)

		# Pool Configuration
		self.pool_seq_length_forward = self.num_players * self.num_forwards
		self.pool_seq_length_defense = self.num_players * self.num_defensemen
		self.pool_seq_length_goalie = self.num_players * self.num_goalies

		# Correct pool projection layers to input_dim=1
		self.pool_forward_proj = nn.Linear(1, 8)
		self.pool_defense_proj = nn.Linear(1, 8)
		self.pool_goalie_proj = nn.Linear(1, 8)

		# Attention Pool Branches
		self.pool_forward = AttentionPoolBranch(self.pool_seq_length_forward)
		self.pool_defense = AttentionPoolBranch(self.pool_seq_length_defense)
		self.pool_goalie = AttentionPoolBranch(self.pool_seq_length_goalie)

		# Merge Layer
		self.merged_fc = nn.Sequential(
			nn.Linear(16+8+8+24, 32),  # Global(16) + Our(8) + Opp(8) + Pools(24)
			nn.BatchNorm1d(32),
			nn.SiLU(),
			nn.Linear(32, 16),
			nn.BatchNorm1d(16),
			nn.SiLU()
		)

		# Heads
		self.policy_head = nn.Sequential(
			nn.Linear(16, 16),
			nn.BatchNorm1d(16),
			nn.SiLU(),
			nn.Linear(16, 2)
		)
		self.value_head = nn.Sequential(
			nn.Linear(16, 16),
			nn.BatchNorm1d(16),
			nn.SiLU(),
			nn.Linear(16, 1)
		)

	def forward(self, state):
		B = state["global"].shape[0]
		
		# Global Features
		global_feat = self.global_branch(state["global"])

		# Our Features
		our_feat = self.our_branch(state["our"])

		# Opponent Features
		opponents = state["opponents"].view(B*(self.num_players-1), -1)
		opp_embeds = self.opp_shared(opponents).view(B, (self.num_players-1)*16)
		opp_feat = self.opponents_branch(opp_embeds)

		# Process Pool Branches
		def process_pool(pool_input, proj_layer, pool_module, seq_length):
			# Reshape to [batch, seq_length, features]
			reshaped = pool_input.view(B, seq_length, -1)
			projected = proj_layer(reshaped)
			return pool_module(projected)

		pool_fwd = process_pool(
			state["pool_forward"],
			self.pool_forward_proj,
			self.pool_forward,
			self.pool_seq_length_forward
		)
		pool_def = process_pool(
			state["pool_defense"],
			self.pool_defense_proj,
			self.pool_defense,
			self.pool_seq_length_defense
		)
		pool_goal = process_pool(
			state["pool_goalie"],
			self.pool_goalie_proj,
			self.pool_goalie,
			self.pool_seq_length_goalie
		)
		pool_feat = torch.cat([pool_fwd, pool_def, pool_goal], dim=1)

		# Merge Features
		merged = torch.cat([global_feat, our_feat, opp_feat, pool_feat], dim=1)
		merged_feat = self.merged_fc(merged)

		# Policy Head
		policy_out = self.policy_head(merged_feat)
		mu = 0.5 * (torch.tanh(policy_out[:, 0]) + 1) * (1 - 2e-3) + 1e-3
		sigma = 0.05 * torch.sigmoid(policy_out[:, 1]) + 1e-3
		a_param = torch.clamp((mu**2 * (1 - mu)) / (sigma**2 + 1e-8) - mu, min=1e-3)
		b_param = torch.clamp(a_param * (1/(mu + 1e-8) - 1), min=1e-3)

		# Value Head
		value = self.value_head(merged_feat).squeeze(-1)

		return a_param, b_param, value
'''

class PPOAuctionEnvNet(nn.Module):
    def __init__(self, game, summary_dim=24, opponent_dim=5, hidden_dim=64):
        """
        A simple network that expects a single concatenated vector:
            summary (24 dims) concatenated with flattened opponent stats (num_opponents * 4 dims).
        """
        super(PPOAuctionEnvNet, self).__init__()
        input_dim = summary_dim + (game.getStructure()["NUM_PLAYERS"] - 1) * opponent_dim
        self.fc = nn.Linear(input_dim, hidden_dim)
        self.common = nn.Sequential(
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        # Policy head: outputs 2 numbers (used to compute mu and sigma)
        self.policy_head = nn.Linear(hidden_dim, 2)
        # Value head: outputs a single scalar value
        self.value_head = nn.Linear(hidden_dim, 1)
    
    def forward(self, state):
        """
        Expects state to be a dictionary with a key "simple" of shape (B, input_dim)
        Processes the vector and returns:
            a_param, b_param, value
        following the original operations.
        """
        x = state["simple"]
        x = self.fc(x)
        x = self.common(x)
        
        # Policy Head with operations
        policy_out = self.policy_head(x)
        mu = 0.5 * (torch.tanh(policy_out[:, 0]) + 1) * (1 - 2e-3) + 1e-3
        sigma = 0.05 * torch.sigmoid(policy_out[:, 1]) + 1e-3
        a_param = torch.clamp((mu**2 * (1 - mu)) / (sigma**2 + 1e-8) - mu, min=1e-3)
        b_param = torch.clamp(a_param * (1/(mu + 1e-8) - 1), min=1e-3)
        
        # Value Head
        value = self.value_head(x).squeeze(-1)
        return a_param, b_param, value
	
# -----------------------------------------------------------------------------
# PPO Training Wrapper
# -----------------------------------------------------------------------------

# Note: This wrapper assumes that you have a game object that can prepare batched tensor inputs,
# e.g. via game.prepare_tensor_input(states, device).
# Also, it assumes that you collect trajectories (or batches) of tuples
# (state, action, old_log_prob, return, advantage).
#
# In PPO, the loss is computed as:
#   L = min( r * advantage, clip(r, 1-ε, 1+ε) * advantage )
# where r = exp(new_log_prob - old_log_prob)
# plus a value loss and an entropy bonus.
#
# You may need to adapt the details (e.g. advantage normalization, multiple PPO epochs, etc.)
# to suit your application.
import os
import copy
import torch
import torch.nn as nn
import torch.optim as optim
from torch import distributions
from random import shuffle
from tqdm import tqdm

# Make sure you have these or similar helper classes/functions defined:
# - dotdict
# - AverageMeter
# - Arena

class PPONNetWrapper():  # You can also subclass your base NeuralNet if desired.
	def __init__(self, game):
		self.nnet = PPOAuctionEnvNet(game)
		self.game = game
		# Adjust hyperparameters as desired.
		self.args = dotdict({
			'episodes_per_iter' : 50,
			'entropy_coef' : 1,
			'lambda_gae' : 0.01,
			'critic_coef' : 1.0/100,
			'arenaCompare' : 10,
			'compare_rate' : 1,
			'num_iterations' : 1000,
			'gamma' : 0.0,
			'lr': 0.001,
			'clip_epsilon': 1,
			'max_grad_norm' : 1,
			'epochs': 3,
			'batch_size': 128,
			'n_updates_per_iteration' : 4,
			'device': torch.device(
				"cuda" if torch.cuda.is_available() else
				"mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else
				"cpu"
			)
		})
		self.nnet.to(self.args.device)
		self.optimizer = optim.Adam(self.nnet.parameters(), lr=self.args.lr)
		# Initialize previous network as None.
		self.prev_nnet = None

	def select_action(self, state):
		"""
		Given a single state (as a dict), sample an action from the current policy.
		Returns: action, log_prob, value (all as numbers).
		"""
		input_tensors = self.game.prepare_tensor_input([state], device=self.args.device)
		self.nnet.eval()
		with torch.no_grad():
			a_param, b_param, value = self.nnet(input_tensors)
			beta_dist = distributions.Beta(a_param, b_param)
			action = beta_dist.sample()
			log_prob = beta_dist.log_prob(action)
		return action.cpu().numpy()[0], log_prob.cpu().numpy()[0], value.cpu().numpy()[0]
	
	def select_action_prev(self, state):
		"""
		Given a single state (as a dict), sample an action from the current policy.
		Returns: action, log_prob, value (all as numbers).
		"""
		input_tensors = self.game.prepare_tensor_input([state], device=self.args.device)
		self.prev_nnet.eval()
		with torch.no_grad():
			a_param, b_param, value = self.nnet(input_tensors)
			beta_dist = distributions.Beta(a_param, b_param)
			action = beta_dist.sample()
			log_prob = beta_dist.log_prob(action)
		return action.cpu().numpy()[0], log_prob.cpu().numpy()[0], value.cpu().numpy()[0]

	def evaluate_actions(self, states, actions):
		"""
		Given a batch of states and a tensor of actions,
		returns: log_probs, entropy, and values.
		"""
		states = self.game.prepare_tensor_input(states, device=self.args.device)
		a_params, b_params, values = self.nnet(states)
		dist = distributions.Beta(a_params, b_params)
		log_probs = dist.log_prob(actions)
		entropy = dist.entropy()
		return log_probs, entropy, values

	def rollout(self):
		"""
		Collect a batch of self-play data.
		During self-play, Player 1 uses the current policy and the remaining players use the previous policy.
		Only data from Player 1 (current policy) is returned for training.
		Returns batches of states, actions, log_probs, rewards-to-go, and value estimates.
		"""
		num_players = self.game.getStructure()["NUM_PLAYERS"]
		states, acts, log_probs, rtgs, values = [], [], [], [], []


		for _ in tqdm(range(self.args.episodes_per_iter), desc="Episodes", unit="episode"):
			# Create containers for each player's data during the episode
			loc_states = {i: [] for i in range(num_players)}
			loc_acts = {i: [] for i in range(num_players)}
			loc_log_probs = {i: [] for i in range(num_players)}
			loc_rews = {i: [] for i in range(num_players)}
			loc_values = {i: [] for i in range(num_players)}
			
			state = self.game.getInitState()

			while not self.game.getGameOver(state):
				cur_player = self.game.getCurrentPlayer(state)
				# Record reward for the current state (for the player whose turn it is)
				rew = self.game.getRewards(state)[cur_player]
				#pdb.set_trace()
				loc_rews[cur_player].append(rew)
				
				if cur_player == 0:
					# Use the current policy for Player 1
					action, log_prob, value = self.select_action(state)
				else:
					# Use the previous policy for all other players
					action, log_prob, value = self.select_action_prev(state)
						
				loc_acts[cur_player].append(action)
				loc_log_probs[cur_player].append(log_prob)
				loc_states[cur_player].append(state)
				# For Player 1, detach to avoid unnecessary gradients; for others it doesn't matter as we ignore it later.
				loc_values[cur_player].append(value if cur_player == 0 else value)
				
				state = self.game.getNextState(state, action)
			
			# Process and collect only Player 1's data for training
			player = 0  # Only training on current policy data (Player 1)
			# Optionally remove the initial reward if your game setup requires it
			loc_rews[player].pop(0)
			# Append final reward at terminal state
			loc_rews[player].append(self.game.getRewards(state)[player])
			
			loc_rtgs = []
			discounted_reward = 0
			# Compute rewards-to-go (RTG) for Player 1
			for rew in reversed(loc_rews[player]):
				discounted_reward = rew + discounted_reward * self.args.gamma
				loc_rtgs.insert(0, discounted_reward)
			
			# Append Player 1's data to the overall training set
			states.extend(loc_states[player])
			acts.extend(loc_acts[player])
			log_probs.extend(loc_log_probs[player])
			rtgs.extend(loc_rtgs)
			values.extend(loc_values[player])
		
		data = list(zip(states, acts, log_probs, rtgs, values))
		shuffle(data)
		batch_size = self.args.batch_size
		
		def create_batches(data, batch_size):
			batch_states_list, batch_acts_list, batch_log_probs_list, batch_rtgs_list, batch_values_list = [], [], [], [], []
			for i in range(0, len(data), batch_size):
				batch = data[i:i + batch_size]
				batch_states, batch_acts, batch_log_probs, batch_rtgs, batch_values = zip(*batch)
				batch_states_list.append(batch_states)
				batch_acts_list.append(torch.tensor(batch_acts, dtype=torch.float32).to(self.args.device))
				batch_log_probs_list.append(torch.tensor(batch_log_probs, dtype=torch.float32).to(self.args.device))
				batch_rtgs_list.append(torch.tensor(batch_rtgs, dtype=torch.float32).to(self.args.device))
				batch_values_list.append(torch.tensor(batch_values, dtype=torch.float32).to(self.args.device))
			return batch_states_list, batch_acts_list, batch_log_probs_list, batch_rtgs_list, batch_values_list
		
		return create_batches(data, batch_size)


	def learn(self):
		num_players = self.game.getStructure()["NUM_PLAYERS"]

		# Initialize previous network if it doesn't exist
		if self.prev_nnet is None:
			self.prev_nnet = copy.deepcopy(self.nnet)
			self.prev_nnet.eval()

		for i in range(self.args.num_iterations):
			print(f"On iteration {i}")
			batch_states_list, batch_acts_list, batch_log_probs_list, batch_rtgs_list, batch_values_list = self.rollout()
			batches = zip(batch_states_list, batch_acts_list, batch_log_probs_list, batch_rtgs_list, batch_values_list)
			batches = list(batches)

			epochs = tqdm(range(self.args.epochs), desc="Epochs", unit="epoch")
			for epoch in epochs:
				self.nnet.train()
				actor_losses, critic_losses = 0, 0
				
				for batch_states, batch_acts, batch_log_probs, batch_rtgs, batch_values in batches:
					# Evaluate policy once per batch
					curr_log_probs, entropy, values = self.evaluate_actions(batch_states, batch_acts)
					
					# Compute Generalized Advantage Estimation (GAE)
					advantages = self.compute_gae(batch_rtgs, batch_values)

					ratios = torch.exp(curr_log_probs - batch_log_probs)
					surr1 = ratios * advantages
					surr2 = torch.clamp(ratios, 1 - self.args.clip_epsilon, 1 + self.args.clip_epsilon) * advantages
					actor_loss = (-torch.min(surr1, surr2)).mean()
					critic_loss = nn.MSELoss()(values, batch_rtgs.detach())  # Detach target to avoid unnecessary gradients
					
					actor_losses += actor_loss.item()
					critic_losses += critic_loss.item()

					total_loss = (actor_loss + 
								  self.args.critic_coef * critic_loss + 
								  self.args.entropy_coef * entropy.mean())
					
					self.optimizer.zero_grad()
					total_loss.backward()
					utils.clip_grad_norm_(self.nnet.parameters(), self.args.max_grad_norm)
					self.optimizer.step()
				
				epochs.set_postfix(Loss_actor=actor_losses / len(batch_states_list),
								   Loss_critic=critic_losses / len(batch_states_list))
				
			# Update learning rate (annealing)
			'''
			for param_group in self.optimizer.param_groups:
				param_group['lr'] *= 0.99  # Decay factor
			'''
			
			# Update previous network every iteration for stability
			'''
			self.prev_nnet.load_state_dict(self.nnet.state_dict())
			self.prev_nnet.eval()
			'''
			
			# Periodic evaluation
			if i % self.args.compare_rate == 0:
				self.evaluate_current_vs_previous()
				### What if we only update every so often
				self.prev_nnet.load_state_dict(self.nnet.state_dict())
				self.prev_nnet.eval()

	def compute_gae(self, rewards, values):
		advantages = torch.zeros_like(rewards)
		gae = 0
		for t in reversed(range(len(rewards) - 1)):
			delta = rewards[t] + self.args.gamma * values[t + 1] - values[t]
			gae = delta + self.args.gamma * self.args.lambda_gae * gae
			advantages[t] = gae
		return (advantages - advantages.mean()) / (advantages.std() + 1e-10)

	def evaluate_current_vs_previous(self):
		def current_policy_player(state):
			return self.select_action(state)[0]
		
		def previous_policy_player(state):
			input_tensors = self.game.prepare_tensor_input([state], device=self.args.device)
			self.prev_nnet.eval()
			with torch.no_grad():
				a_param, b_param, _ = self.prev_nnet(input_tensors)
				beta_dist = distributions.Beta(a_param, b_param)
				action = beta_dist.sample()
			return action.cpu().numpy()[0]
		
		players = [current_policy_player, previous_policy_player]
		arena = Arena(players, self.game)
		results = arena.playGames(self.args.arenaCompare, verbose=0.1)
		print("Arena results (Current vs. Previous):", results)


	def save_checkpoint(self, folder='checkpoint', filename='checkpoint.pth.tar'):
		filepath = os.path.join(folder, filename)
		if not os.path.exists(folder):
			os.makedirs(folder)
		torch.save({'state_dict': self.nnet.state_dict()}, filepath)

	def load_checkpoint(self, folder='checkpoint', filename='checkpoint.pth.tar'):
		filepath = os.path.join(folder, filename)
		if not os.path.exists(filepath):
			raise Exception("No model in path {}".format(filepath))
		checkpoint = torch.load(filepath, map_location=self.args.device)
		self.nnet.load_state_dict(checkpoint['state_dict'])



'''
class PPONNetWrapper():  # You can also subclass your base NeuralNet if desired.
	def __init__(self, game):
		self.nnet = PPOAuctionEnvNet(game)
		self.game = game
		# Adjust hyperparameters as desired.
		self.args = dotdict({
			'episodes_per_iter' : 100,
			'arenaCompare' : 3,
			'num_iterations' : 1000,
			'gamma' : 0.99,
			'lr': 0.001,
			'clip_epsilon': 0.2,
			'epochs': 4,
			'batch_size': 64,
			'n_updates_per_iteration' : 4,
			'device': torch.device(
				"cuda" if torch.cuda.is_available() else
				"mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else
				"cpu"
			)
		})
		self.nnet.to(self.args.device)
		self.optimizer = optim.Adam(self.nnet.parameters(), lr=self.args.lr)
	
	def select_action(self, state):
		"""
		Given a single state (as a dict), sample an action from the current policy.
		Returns: action, log_prob, value (all as numbers).
		"""
		# Prepare input (assumes game.prepare_tensor_input returns a dict of tensors)
		input_tensors = self.game.prepare_tensor_input([state], device=self.args.device)
		self.nnet.eval()

		with torch.no_grad():
			# Forward pass to get alpha and beta parameters for the Beta distribution
			a_param, b_param, value = self.nnet(input_tensors)
			
			# Create a Beta distribution using the parameters
			beta_dist = distributions.Beta(a_param, b_param)
			
			# Sample an action from the Beta distribution
			action = beta_dist.sample()

			# Calculate log probability for the sampled action
			log_prob = beta_dist.log_prob(action)

		# Return action, log probability, and value
		return action.cpu().numpy()[0], log_prob.cpu().numpy()[0], value.cpu().numpy()[0]
	
	def evaluate_actions(self, states, actions):
		"""
		Given a batch of states and a tensor of actions,
		returns: log_probs, entropy, and values.
		"""

		states = self.game.prepare_tensor_input(states, device=self.args.device)
		a_params, b_params, values = self.nnet(states)
		dist = distributions.Beta(a_params, b_params)
		log_probs = dist.log_prob(actions)
		entropy = dist.entropy()
		return log_probs, entropy, values
	
	def rollout(self):
		"""
		Need to collect a fresh batch of data each time we iterate the actor/critic networks.

		Parameters:
			None

		Return:
			batch_obs - the observations collected this batch. Shape: (number of timesteps, dimension of observation)
			batch_acts - the actions collected this batch. Shape: (number of timesteps, dimension of action)
			batch_log_probs - the log probabilities of each action taken this batch. Shape: (number of timesteps)
			batch_rtgs - the Rewards-To-Go of each timestep in this batch. Shape: (number of timesteps)
		"""

		

		num_players = self.game.getStructure()["NUM_PLAYERS"]

		states = []
		acts = []
		log_probs = []
		rtgs = []

		# Wrap the outer loop with tqdm for progress bar
		for _ in tqdm(range(self.args.episodes_per_iter), desc="Episodes", unit="episode"):
			#pdb.set_trace()
			loc_states = {i : [] for i in range(num_players)}
			loc_acts = {i : [] for i in range(num_players)}
			loc_log_probs = {i : [] for i in range(num_players)}
			loc_rews = {i : [] for i in range(num_players)}
			state = self.game.getInitState()

			# Wrap the inner while loop with tqdm for step progress within an episode
			while not self.game.getGameOver(state):
				cur_player = self.game.getCurrentPlayer(state)

				# Capturing reward of previous move
				rew = self.game.getRewards(state)[cur_player]
				loc_rews[cur_player].append(rew)

				action, log_prob, value = self.select_action(state)

				loc_acts[cur_player].append(action)
				loc_log_probs[cur_player].append(log_prob)
				loc_states[cur_player].append(state)

				state = self.game.getNextState(state, action)

				# Update the progress bar for each step in the game
				#tqdm.write(f"Player {cur_player} took an action")

			for player in range(num_players):
				loc_rews[player].pop(0)

				loc_rtgs = []
				discounted_reward = 0  # The discounted reward so far
				# Iterate through all rewards in the episode. We go backwards for smoother calculation of each
				# discounted return (think about why it would be harder starting from the beginning)
				for rew in reversed(loc_rews[player]):
					discounted_reward = rew + discounted_reward * self.args.gamma
					loc_rtgs.insert(0, discounted_reward)

				states.extend(loc_states[player])
				acts.extend(loc_acts[player])
				log_probs.extend(loc_log_probs[player])
				rtgs.extend(loc_rtgs)

		data = list(zip(states, acts, log_probs, rtgs))
		shuffle(data)

		batch_size = self.args.batch_size

		# Function to create batches manually
		def create_batches(data, batch_size):
			batch_states_list = []
			batch_acts_list = []
			batch_log_probs_list = []
			batch_rtgs_list = []

			# Split the data into batches and add a tqdm progress bar
			for i in range(0, len(data), batch_size):
				batch = data[i:i + batch_size]

				# Unzip the batch into states, acts, log_probs, and rtgs
				batch_states, batch_acts, batch_log_probs, batch_rtgs = zip(*batch)

				# Convert them into torch tensors if needed
				batch_states_list.append(batch_states)

		
				batch_acts_list.append(torch.tensor(batch_acts, dtype=torch.float32).to(self.args.device))
				batch_log_probs_list.append(torch.tensor(batch_log_probs, dtype=torch.float32).to(self.args.device))
				batch_rtgs_list.append(torch.tensor(batch_rtgs, dtype=torch.float32).to(self.args.device))

			return batch_states_list, batch_acts_list, batch_log_probs_list, batch_rtgs_list

		# Example usage of batching
		batch_states_list, batch_acts_list, batch_log_probs_list, batch_rtgs_list = create_batches(data, batch_size)

		return batch_states_list, batch_acts_list, batch_log_probs_list, batch_rtgs_list


	def learn(self):
		num_players = self.game.getStructure()["NUM_PLAYERS"]
		for i in range(self.args.num_iterations):
			print(f"On iteration {i}")
			states, acts, log_probs, rtgs = self.rollout()  

			batches = zip(states, acts, log_probs, rtgs)

			epochs = tqdm(range(self.args.epochs), desc="Epochs", unit="epoch")
			for epoch in epochs:
				actor_losses = AverageMeter()
				critic_losses = AverageMeter()
				for batch in batches:
					batch_states, batch_acts, batch_log_probs, batch_rtgs = batch
					# Calculate advantage at k-th iteration
					log_probs, entropy, values = self.evaluate_actions(batch_states, batch_acts)
					A_k = batch_rtgs - values.detach()                                                                       # ALG STEP 5

					# One of the only tricks I use that isn't in the pseudocode. Normalizing advantages
					# isn't theoretically necessary, but in practice it decreases the variance of 
					# our advantages and makes convergence much more stable and faster. I added this because
					# solving some environments was too unstable without it.
					A_k = (A_k - A_k.mean()) / (A_k.std() + 1e-10)

					# This is the loop where we update our network for some n epochs
					for _ in range(self.args.n_updates_per_iteration):                                                       # ALG STEP 6 & 7
						# Calculate V_phi and pi_theta(a_t | s_t)
						curr_log_probs, entropy, values = self.evaluate_actions(batch_states, batch_acts)

						# Calculate the ratio pi_theta(a_t | s_t) / pi_theta_k(a_t | s_t)
						# NOTE: we just subtract the logs, which is the same as
						# dividing the values and then canceling the log with e^log.
						# For why we use log probabilities instead of actual probabilities,
						# here's a great explanation: 
						# https://cs.stackexchange.com/questions/70518/why-do-we-use-the-log-in-gradient-based-reinforcement-algorithms
						# TL;DR makes gradient ascent easier behind the scenes.
						ratios = torch.exp(curr_log_probs - batch_log_probs)

						# Calculate surrogate losses.
						surr1 = ratios * A_k
						surr2 = torch.clamp(ratios, 1 - self.args.clip_epsilon, 1 + self.args.clip_epsilon) * A_k

						# Calculate actor and critic losses.
						# NOTE: we take the negative min of the surrogate losses because we're trying to maximize
						# the performance function, but Adam minimizes the loss. So minimizing the negative
						# performance function maximizes it.
						actor_loss = (-torch.min(surr1, surr2)).mean()
						critic_loss = nn.MSELoss()(values, batch_rtgs)

						actor_losses.update(actor_loss, self.args.batch_size)
						critic_losses.update(critic_loss, self.args.batch_size)

						epochs.set_postfix(Loss_actor=actor_losses, Loss_critic=critic_losses)

						total_loss = actor_loss + 0.01*critic_loss

						# compute gradient and do SGD step
						self.optimizer.zero_grad()
						total_loss.backward()
						self.optimizer.step()

			if i % 10 == 0:
				def policy_player(state):
					return self.select_action(state)[0]
				
				players = [policy_player for i in range(num_players)]
				arena = Arena(players, self.game)

				results = arena.playGames(self.args.arenaCompare)
				print(results)
			

	
	def save_checkpoint(self, folder='checkpoint', filename='checkpoint.pth.tar'):
		filepath = os.path.join(folder, filename)
		if not os.path.exists(folder):
			os.makedirs(folder)
		torch.save({'state_dict': self.nnet.state_dict()}, filepath)
	
	def load_checkpoint(self, folder='checkpoint', filename='checkpoint.pth.tar'):
		filepath = os.path.join(folder, filename)
		if not os.path.exists(filepath):
			raise Exception("No model in path {}".format(filepath))
		checkpoint = torch.load(filepath, map_location=self.args.device)
		self.nnet.load_state_dict(checkpoint['state_dict'])

'''