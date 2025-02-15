import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb
def compute_stats(lst):
	"""
	Given a list of numbers (e.g. athlete means),
	compute summary statistics:
	  [number_left, mean, max, 25th percentile, 75th percentile].
	"""
	if len(lst) == 0:
		return [0.0, 0.0, 0.0, 0.0, 0.0]
	num = float(len(lst))
	mean = np.mean(lst)
	max_val = np.max(lst)
	p25 = np.percentile(lst, 25)
	p75 = np.percentile(lst, 75)
	return [num, mean, max_val, p25, p75]


def prepare_batch_state(batch_obs, device="cpu"):
	"""
	Prepares a simplified state dictionary with one key "simple"
	that concatenates a summary vector and flattened opponent stats.
	
	For each game state, builds:
	  - Summary (24 dims):
			[forward_stats (5), defense_stats (5), goalie_stats (5),
			 nominated_player_mean (1), nominated_player_onehot (3),
			 our_budget, our_mean, our_forwards_needed, our_defense_needed, our_goalies_needed]
	  - Opponents: For each opponent, a 5-dim vector:
			[their_mean, their_forwards_needed, their_defense_needed, 
			 their_goalies_needed, their_budget]
	
	The final concatenated vector has dimension:
		 24 + (number_of_opponents * 5)
		 
	In this version we “normalize” various quantities:
	   - For athlete stats, we normalize the count by NUM_PLAYERS and
		 the performance metrics by the expected gamma mean (shape * scale).
	   - For the nominated player's mean, we use the appropriate gamma mean.
	   - For budget values, we divide by GAME_BUDGET.
	   - For needed counts (both ours and opponents), we divide by NUM_PLAYERS.
	   - For our_mean and opponents’ means (which come from our team stats),
		 we normalize by the average gamma mean.
	"""
	
	def normalize_stats(stats, max_count, gamma_shape, gamma_scale):
		# stats: [count, mean, max, p25, p75]
		# Normalize count by max_count and other metrics by the gamma expected mean.
		gamma_mean = gamma_shape * gamma_scale
		norm_count = stats[0] / max_count
		norm_metrics = [s / gamma_mean for s in stats[1:]]
		return [norm_count] + norm_metrics

	summary_list = []
	opponents_list = []
	
	for state in batch_obs:
		# Raw state variables
		num_players = state["NUM_PLAYERS"]
		start_budget = state["GAME_BUDGET"]
		forwards_needed = state["FORWARDS_NEEDED"]
		defense_needed = state["DEFENSEMEN_NEEDED"]
		goalies_needed = state["GOALIES_NEEDED"]

		# Gamma parameters for athlete performance distributions
		f_shape, f_scale = state["FORWARDS_SHAPE"], state["FORWARDS_SCALE"]
		d_shape, d_scale = state["DEFENSE_SHAPE"], state["DEFENSE_SCALE"]
		g_shape, g_scale = state["GOALIES_SHAPE"], state["GOALIES_SCALE"]

		# --- Compute athlete summary stats for each position ---
		f_stats = compute_stats(state["athletes_left"]["forward"])
		d_stats = compute_stats(state["athletes_left"]["defenseman"])
		g_stats = compute_stats(state["athletes_left"]["goalie"])
		
		# Normalize athlete stats using gamma parameters.
		f_stats_norm = normalize_stats(f_stats, forwards_needed * num_players, f_shape, f_scale)
		d_stats_norm = normalize_stats(d_stats, defense_needed * num_players, d_shape, d_scale)
		g_stats_norm = normalize_stats(g_stats, goalies_needed * num_players, g_shape, g_scale)
		
		# --- Nominated player info ---
		nominated_mean = state["nominated_player"]["mean"]
		nominated_pos = state["nominated_player"]["position"].lower()
		if nominated_pos == "forward":
			gamma_for_nom = f_shape * f_scale
			nominated_onehot = [1.0, 0.0, 0.0]
		elif nominated_pos == "defenseman":
			gamma_for_nom = d_shape * d_scale
			nominated_onehot = [0.0, 1.0, 0.0]
		elif nominated_pos == "goalie":
			gamma_for_nom = g_shape * g_scale
			nominated_onehot = [0.0, 0.0, 1.0]
		else:
			gamma_for_nom = 1.0  # fallback to avoid division by zero
			nominated_onehot = [0.0, 0.0, 0.0]
		
		norm_nominated_mean = nominated_mean / gamma_for_nom
		
		# --- Our team info ---
		pid = state["PLAYER_ID"]
		our_budget = state["budgets"][pid]
		our_mean = state["members_means"][pid]
		our_forwards_needed = state["members_forwards_needed"][pid]
		our_defense_needed = state["member_defense_needed"][pid]
		our_goalies_needed = state["member_goalies_needed"][pid]
		
		# Normalize our team info.
		norm_our_budget = our_budget / start_budget
		
		# For team member means (ours and opponents) we use an average gamma mean.
		avg_gamma_mean = (f_shape * f_scale * forwards_needed + d_shape * d_scale * defense_needed + g_shape * g_scale * goalies_needed) / (forwards_needed + defense_needed + goalies_needed)
		norm_our_mean = our_mean / avg_gamma_mean
		
		norm_our_forwards_needed = our_forwards_needed / forwards_needed
		norm_our_defense_needed = our_defense_needed / defense_needed
		norm_our_goalies_needed = our_goalies_needed / goalies_needed
		
		# --- Build the 24-dim summary vector ---
		summary = (
			f_stats_norm + d_stats_norm + g_stats_norm +
			[norm_nominated_mean] + nominated_onehot +
			[norm_our_budget, norm_our_mean, norm_our_forwards_needed,
			 norm_our_defense_needed, norm_our_goalies_needed]
		)
		summary_list.append(summary)
		
		# --- Build opponent stats ---
		opp_stats = []
		for i, (opp_mean, opp_fneed, opp_dneed, opp_gneed, opp_budget) in enumerate(
			zip(state["members_means"],
				state["members_forwards_needed"],
				state["member_defense_needed"],
				state["member_goalies_needed"],
				state["budgets"])
		):
			if i == pid:
				continue  # skip our own stats
			# Normalize opponent info:
			norm_opp_mean = opp_mean / avg_gamma_mean
			norm_opp_fneed = opp_fneed / forwards_needed
			norm_opp_dneed = opp_dneed / defense_needed
			norm_opp_gneed = opp_gneed / goalies_needed
			norm_opp_budget = opp_budget / start_budget
			opp_stats.append([norm_opp_mean, norm_opp_fneed, norm_opp_dneed,
							  norm_opp_gneed, norm_opp_budget])
		opponents_list.append(opp_stats)
	
	summary_tensor = torch.tensor(summary_list, dtype=torch.float32, device=device)
	opponents_tensor = torch.tensor(opponents_list, dtype=torch.float32, device=device)
	B = summary_tensor.shape[0]
	opponents_flat = opponents_tensor.view(B, -1)
	simple_tensor = torch.cat([summary_tensor, opponents_flat], dim=1)

	#pdb.set_trace()
	
	return {"simple": simple_tensor}



# Example usage:
if __name__ == "__main__":
	# Suppose we have a batch of game state dictionaries
	batch_obs = [
		# Each dictionary should include keys:
		# "athletes_left", "nominated_player", "PLAYER_ID", "budgets",
		# "members_means", "members_forwards_needed", "member_defense_needed", "member_goalies_needed"
		# For example:
		{
			"athletes_left": {
				"forward": [70, 75, 80],
				"defenseman": [60, 65, 68],
				"goalie": [55, 60]
			},
			"nominated_player": {"mean": 78, "position": "Forward"},
			"PLAYER_ID": 0,
			"budgets": [100, 90, 80, 70],
			"members_means": [75, 70, 65, 60],
			"members_forwards_needed": [2, 1, 1, 2],
			"member_defense_needed": [2, 1, 2, 1],
			"member_goalies_needed": [1, 1, 0, 1],
		},
		# Add additional state dicts for a full batch as needed.
	]
	
	device = "cpu"
	state = prepare_simple_batch_state(batch_obs, device=device)
	
	# Assume 4 players total, so there are 3 opponents per state.
	num_opponents = 3  
	net = SimpleAuctionNet(num_opponents=num_opponents).to(device)
	
	a_param, b_param, value = net(state)
	print("a_param:", a_param)
	print("b_param:", b_param)
	print("Value:", value)
