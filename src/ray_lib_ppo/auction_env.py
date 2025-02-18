import numpy as np
from enum import Enum
from scipy.stats import gamma
import random
import copy
from ray.rllib.env.multi_agent_env import MultiAgentEnv
import gymnasium as gym
import numpy as np
import json
import pdb
NUM_SIMS = 10000
shape = 2.5
scale = 20
INTERMEDIATE_COEF = 0.1



#######################
# RLlib-compatible Auction Environment
#######################

class AuctionEnv(MultiAgentEnv):
	"""
	This environment implements a Vickrey auction for a fantasy hockey draft.
	Instead of sequential bidding, all agents submit their bids simultaneously.

	League Structure:
	  - N teams (players), each with a fixed budget.
	  - In each auction round a player (athlete) is nominated.
	  - Every team submits a bid simultaneously.
	  - The highest bid wins, but the winner pays the second-highest bid.

	The environment tracks budgets, roster needs, and the history of auctions.
	"""

	class Position(Enum):
		GOALIE = "goalie"
		FORWARD = "forward"
		DEFENSEMAN = "defenseman"

	class Athlete:
		def __init__(self, mean, position, owner=-1):
			self.mean = mean
			self.position = position
			self.owner = owner
			pos_code = "G"
			if self.position == AuctionEnv.Position.FORWARD:
				pos_code = "F"
			elif self.position == AuctionEnv.Position.DEFENSEMAN:
				pos_code = "D"
			self.hash = f"{pos_code}-{self.mean}"

		def has_owner(self):
			return self.owner >= 0

		def set_owner(self, owner):
			self.owner = owner

		def __repr__(self):
			if self.owner >= 0:
				return f"Athlete(mean={self.mean}, position={self.position}, owner={self.owner})"
			else:
				return f"Athlete(mean={self.mean}, position={self.position})"

		def __hash__(self):
			return hash(self.hash)

		def __eq__(self, other):
			return isinstance(other, AuctionEnv.Athlete) and (self.hash == other.hash)

		def __lt__(self, other):
			if self.position != other.position:
				order = {AuctionEnv.Position.FORWARD: 0,
						 AuctionEnv.Position.DEFENSEMAN: 1,
						 AuctionEnv.Position.GOALIE: 2}
				return order[self.position] < order[other.position]
			return self.mean < other.mean

	@staticmethod
	def generate_athletes(num_players, num_forwards, num_defensemen, num_goalies, shape, scale):
		"""
		Generate a list of athletes for the auction.
		"""
		athletes = []
		for _ in range(num_forwards * num_players):
			mean = np.round(gamma.rvs(a=shape, scale=scale), 2)
			mean = min(250, mean)
			athletes.append(AuctionEnv.Athlete(mean=mean, position=AuctionEnv.Position.FORWARD))
			
		for _ in range(num_defensemen * num_players):
			mean = np.round(gamma.rvs(a=shape, scale=scale * 0.55), 2)
			mean = min(250, mean)
			athletes.append(AuctionEnv.Athlete(mean=mean, position=AuctionEnv.Position.DEFENSEMAN))
		for _ in range(num_goalies * num_players):
			mean = np.round(gamma.rvs(a=shape, scale=scale), 2)
			mean = min(250, mean)
			athletes.append(AuctionEnv.Athlete(mean=mean, position=AuctionEnv.Position.GOALIE))

		return athletes
	@staticmethod
	def generate_nomination_order(athletes):
		# Separate athletes by position
		forwards = sorted([a for a in athletes if a.position == AuctionEnv.Position.FORWARD], key=lambda x: -1 * x.mean)
		defensemen = sorted([a for a in athletes if a.position == AuctionEnv.Position.DEFENSEMAN], key=lambda x: -1 * x.mean)
		goalies = sorted([a for a in athletes if a.position == AuctionEnv.Position.GOALIE], key=lambda x: -1 * x.mean)

		# Round-robin selection
		nomination_order = []
		pools = [forwards, defensemen, goalies]

		while any(pools):
			# Weight pools by their length
			pool_weights = [len(pool) for pool in pools]

			# Select a pool based on weighted length
			pool = random.choices(pools, weights=pool_weights, k=1)[0]

			# Append the first athlete from the selected pool and remove them
			nomination_order.append(pool.pop(0))

		return nomination_order

	def __init__(self, num_players=2, budget=1000, num_forwards=10, num_defensemen=5, num_goalies=2, state=None):
		super().__init__()
		# For simplicity, we assume state=None (i.e. fresh game)
		self.agents = [f"agent_{i}" for i in range(num_players)]
		if state:
			# If loading from a state dictionary, rebuild the environment.
			# (Your original code for state restoration would go here.)
			pass
		else:
			self.num_players = num_players
			self.GAME_BUDGET = budget
			self.FORWARDS_NEEDED = num_forwards
			self.DEFENSEMEN_NEEDED = num_defensemen
			self.GOALIES_NEEDED = num_goalies

			# Create athletes and shuffle their order.
			self.athletes = AuctionEnv.generate_athletes(
				self.num_players, self.FORWARDS_NEEDED, self.DEFENSEMEN_NEEDED, self.GOALIES_NEEDED, shape, scale)
			self.archive_athletes = self.athletes.copy()
			#random.shuffle(self.athletes)
			self.athletes = AuctionEnv.generate_nomination_order(self.athletes)

			# Sanity checks
			empiric_forwards = np.sum([athlete.position == self.Position.FORWARD for athlete in self.athletes])
			empiric_goalies = np.sum([athlete.position == self.Position.GOALIE for athlete in self.athletes])
			empiric_defense = np.sum([athlete.position == self.Position.DEFENSEMAN for athlete in self.athletes])
			assert empiric_forwards == num_forwards * self.num_players, "Mismatch in number of forwards"
			assert empiric_goalies == num_goalies * self.num_players, "Mismatch in number of goalies"
			assert empiric_defense == num_defensemen * self.num_players, "Mismatch in number of defensemen"

			self.forwards_left = self.num_players * self.FORWARDS_NEEDED
			self.defense_left = self.num_players * self.DEFENSEMEN_NEEDED
			self.goalies_left = self.num_players * self.GOALIES_NEEDED
			self.rounds_left = len(self.athletes)

			self.budgets = np.ones(self.num_players) * self.GAME_BUDGET
			# Array to hold bids from each agent (initialized to -1)
			self.bids_placed = -1 * np.ones(self.num_players)
			# Requirements for each team (arrays of length num_players)
			self.members_forwards_needed = np.ones(self.num_players) * self.FORWARDS_NEEDED
			self.member_defense_needed = np.ones(self.num_players) * self.DEFENSEMEN_NEEDED
			self.member_goalies_needed = np.ones(self.num_players) * self.GOALIES_NEEDED

			# Start with a nominated player (pop from list)
			self.nominated_player = self.athletes.pop(0)

			self.members_means = np.zeros(self.num_players)
			self.last_winner = -1
			self.last_price = -1
			self.prev_bids = np.zeros(self.num_players)

			self.game_over = False
			self.history = {}  # Auction history

			self.illegal_bid_penalties = np.zeros(self.num_players)

		# Define the action space for each agent.
		per_agent_action_space = gym.spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)
		self.action_space = gym.spaces.Dict({
			agent: per_agent_action_space for agent in self.agents
		})

		# Define an observation space. (You’ll need to adjust the bounds/shapes to your needs.)
		obs_dim = (
			1 +  # Reference budget scale
			1 + # must bid or not
			1 + # max bid
			#1 +  # Agent ID (normalized)
			num_players +  # Normalized player means
			num_players +  # Normalized budgets
			num_players +  # Normalized forwards needed
			num_players +  # Normalized defensemen needed
			num_players +  # Normalized goalies needed
			1 +  # Normalized forwards left
			1 +  # Normalized defense left
			1 +  # Normalized goalies left
			1 +  # Nominated player mean
			3 +  # One-hot position encoding for nominated player
			#self.num_players + #last set of bids?
			self.FORWARDS_NEEDED * num_players+  # Normalized forward values
			self.DEFENSEMEN_NEEDED * num_players +  # Normalized defense values
			self.GOALIES_NEEDED * num_players  # Normalized goalie values
		)

		per_agent_obs_space = gym.spaces.Box(low=0.0, high=1.0, shape=(obs_dim,), dtype=np.float32)
		self.observation_space = gym.spaces.Dict({
			agent: per_agent_obs_space for agent in self.agents
		})

	# --- Helper Methods for Simultaneous Bidding ---

	def get_must_bid(self, agent):
		"""
		Returns 1 if the agent is forced to bid on the current nominated player,
		based on their remaining roster needs.
		"""
		if (self.nominated_player.position == self.Position.FORWARD) and (self.forwards_left == self.members_forwards_needed[agent]):
			return 1
		elif (self.nominated_player.position == self.Position.DEFENSEMAN) and (self.defense_left == self.member_defense_needed[agent]):
			return 1
		elif (self.nominated_player.position == self.Position.GOALIE) and (self.goalies_left == self.member_goalies_needed[agent]):
			return 1
		return 0

	def get_max_bid(self, player):
		"""
		Returns the maximum bid an agent can make.
		(Here we use a fixed range, e.g., 0 to 1000.)
		"""
		if self.nominated_player.position == self.Position.FORWARD:
			if self.members_forwards_needed[player] == 0:
				return 0
		elif self.nominated_player.position == self.Position.DEFENSEMAN:
			if self.member_defense_needed[player] == 0:
				return 0
		elif self.nominated_player.position == self.Position.GOALIE:
			if self.member_goalies_needed[player] == 0:
				return 0

		max_val = self.budgets[player] - self.members_forwards_needed[player] - self.member_defense_needed[player] - self.member_goalies_needed[player] + 1
		return max_val

	def sanitize_bid(self, bid, agent):
		"""
		Optionally adjust the bid probabilities (or bid value) if the agent is forced.
		"""
		# For example, if must bid then ensure bid is raised slightly.
		if self.get_must_bid(agent):
			bid += 1e-2
		return bid

	# --- RLlib Required Methods: reset and step ---

	def reset(self, seed=None, options=None):
		"""
		Reset the environment to the initial state and return observations for all agents.
		"""
		# For simplicity, we reinitialize a new game.
		self.__init__(num_players=self.num_players, budget=self.GAME_BUDGET,
					  num_forwards=self.FORWARDS_NEEDED, num_defensemen=self.DEFENSEMEN_NEEDED,
					  num_goalies=self.GOALIES_NEEDED)
		obs = {}
		for i in range(self.num_players):
			obs[f"agent_{i}"] = self.get_observation_for_agent(i)
		return obs, {}

	def get_game_log(self):
		output = ""
		output += f"Final Means: {self.members_means}\n"
		output += f"Final Budgets: {self.budgets}\n"
		output += "\n\n\nHistory:\n"
		for x in self.history:
			output += f"{str(x)} : {str(self.history[x])}\n"
		return output


	def step(self, action_dict):
		"""
		Expects a dictionary of actions from all agents.
		Each agent's action is assumed to be a continuous value in [0, 1] that
		is scaled to the bid range [min_bid, max_bid].
		Processes all bids simultaneously and advances the auction by one nominated player.
		"""
		#pdb.set_trace()
		#print(action_dict)
		# Process bids for each agent.
		for agent_id, action in action_dict.items():
			#action = action_dict["action"]
			# Extract agent index from string (assumes "agent_0", "agent_1", etc.)
			agent_index = int(agent_id.split("_")[1])
			min_bid = self.get_must_bid(agent_index)
			max_bid = self.get_max_bid(agent_index)

			bid = action * self.GAME_BUDGET

			self.illegal_bid_penalties[agent_index] = min(0, bid - min_bid, max_bid - bid)

			bid = np.clip(bid, min_bid, max_bid)
			# Scale the action to the bid range.
			#bid = action * (max_bid - min_bid) + min_bid
			bid = np.round(bid)
			self.bids_placed[agent_index] = bid

		# Determine the winner using your Vickrey auction rule.
		scores = self.bids_placed
		max_score = np.max(scores)
		winners = np.where(scores == max_score)[0]
		winner = np.random.choice(winners)
		# Second-highest bid: use partitioning.
		if len(self.bids_placed) > 1:
			price_paid = np.partition(self.bids_placed, -2)[-2]
		else:
			price_paid = self.bids_placed[0]

		self.last_winner = winner
		self.last_price = price_paid
		self.prev_bids = copy.deepcopy(list(self.bids_placed))
		#print("storing old bids:", self.prev_bids)

		# Update winner's budget and history.
		self.budgets[winner] -= price_paid
		self.history[self.nominated_player] = {"price": price_paid, "winner": winner, "bids" : copy.deepcopy(self.prev_bids), "cur_budgets" : copy.deepcopy(list(self.budgets))}
		self.nominated_player.set_owner(winner)
		self.members_means[winner] += self.nominated_player.mean

		# Update roster needs based on the position.
		if self.nominated_player.position == self.Position.FORWARD:
			self.members_forwards_needed[winner] -= 1
			self.forwards_left -= 1
		elif self.nominated_player.position == self.Position.DEFENSEMAN:
			self.member_defense_needed[winner] -= 1
			self.defense_left -= 1
		elif self.nominated_player.position == self.Position.GOALIE:
			self.member_goalies_needed[winner] -= 1
			self.goalies_left -= 1

		# Reset bids for the next round.
		self.bids_placed = -1 * np.ones(self.num_players)

		# Advance to the next nominated player.
		if len(self.athletes) == 0:
			self.game_over = True
		else:
			self.nominated_player = self.athletes.pop(0)

		# Build observations for all agents.
		obs = {}
		for i in range(self.num_players):
			obs[f"agent_{i}"] = self.get_observation_for_agent(i)

		# Get rewards—here we use your get_reward method.
		rewards_arr = self.get_reward()
		rewards = {}
		for i in range(self.num_players):
			rewards[f"agent_{i}"] = rewards_arr[i]

		# The environment is done if the game is over.
		dones = {"__all__": self.game_over}
		truncated = {"__all__" : False}
		info = {}
		return obs, rewards, dones, truncated, info

	def get_human_obs(self, agent):
		# Pre-sort athletes by their mean for each position
		forwards = sorted(
			[athlete.mean for athlete in self.athletes if athlete.position == self.Position.FORWARD],
			reverse=True
		)
		defensemen = sorted(
			[athlete.mean for athlete in self.athletes if athlete.position == self.Position.DEFENSEMAN],
			reverse=True
		)
		goalies = sorted(
			[athlete.mean for athlete in self.athletes if athlete.position == self.Position.GOALIE],
			reverse=True
		)

		# Format history
		history_str = "\n".join([
			f"  {athlete.hash}: Price {info['price']} | Winner: {info['winner']} | Bids: {info['bids']} | Budgets: {info['cur_budgets']}"
			for athlete, info in self.history.items()
		])

		table = f"""
	======================== GAME STATE ========================
	Agent            : {agent}
	============================================================
	AUCTION HISTORY
	{history_str or "  No auction history yet."}
	============================================================
	ROSTER NEEDS
		Forwards   : {self.members_forwards_needed}
		Defensemen : {self.member_defense_needed}
		Goalies    : {self.member_goalies_needed}
	------------------------------------------------------------
	ATHLETES
		Forwards   : {', '.join(str(val) for val in forwards)}
		Defensemen : {', '.join(str(val) for val in defensemen)}
		Goalies    : {', '.join(str(val) for val in goalies)}
	------------------------------------------------------------
	Budgets          : {self.budgets}
	Members Means    : {self.members_means}
	------------------------------------------------------------
	NOMINATED PLAYER
		Mean       : {self.nominated_player.mean}
		Position   : {self.nominated_player.position}
	------------------------------------------------------------

	"""
		return table


	# --- Observation and Reward Helpers ---


	def get_observation_for_agent(self, agent):
		"""
		Returns a normalized, flattened observation vector for the given agent.
		"""
		def one_hot_encode_pos(pos):
			if pos == self.Position.FORWARD:
				return [1, 0, 0]
			elif pos == self.Position.DEFENSEMAN:
				return [0, 1, 0]
			elif pos == self.Position.GOALIE:
				return [0, 0, 1]
			return [0, 0, 0]  # Default case

		# Normalize budgets (assuming max budget is GAME_BUDGET)
		normalized_budgets = np.array(self.budgets) / self.GAME_BUDGET

		# Normalize player means (assuming max player value ~ 100)
		max_player_value = 400  # Can be adjusted based on distribution
		normalized_means = np.array(self.members_means) / max_player_value
		nominated_player_mean = self.nominated_player.mean / max_player_value

		# Normalize roster needs (dividing by max players needed)
		max_team_size = self.FORWARDS_NEEDED + self.DEFENSEMEN_NEEDED + self.GOALIES_NEEDED
		normalized_forwards_needed = np.array(self.members_forwards_needed) / self.FORWARDS_NEEDED
		normalized_defense_needed = np.array(self.member_defense_needed) / self.DEFENSEMEN_NEEDED
		normalized_goalies_needed = np.array(self.member_goalies_needed) / self.GOALIES_NEEDED

		# Normalize remaining players by max pool size
		max_forwards = self.FORWARDS_NEEDED * self.num_players
		max_defense = self.DEFENSEMEN_NEEDED * self.num_players
		max_goalies = self.GOALIES_NEEDED * self.num_players

		normalized_forwards_left = self.forwards_left / max_forwards
		normalized_defense_left = self.defense_left / max_defense
		normalized_goalies_left = self.goalies_left / max_goalies

		# bid restrictions
		min_bid = self.get_must_bid(agent)
		max_bid = self.get_max_bid(agent)/self.GAME_BUDGET

		# Normalize remaining players' values and pad
		forwards = sorted(
			[athlete.mean / max_player_value for athlete in self.athletes if athlete.position == self.Position.FORWARD],
			reverse=True
		)
		forwards = forwards+ [0] * (max_forwards - len(forwards))

		defensemen = sorted(
			[athlete.mean / max_player_value for athlete in self.athletes if athlete.position == self.Position.DEFENSEMAN],
			reverse=True
		)
		defensemen = defensemen + [0] * (max_defense - len(defensemen))

		goalies = sorted(
			[athlete.mean / max_player_value for athlete in self.athletes if athlete.position == self.Position.GOALIE],
			reverse=True
		)
		goalies = goalies + [0] * (max_goalies - len(goalies))

		# One-hot encode nominated player position
		nominated_pos_one_hot = one_hot_encode_pos(self.nominated_player.position)

		#prev bids - normlaized to 0,1
		prev = np.array(copy.deepcopy(self.prev_bids))
		prev /= self.GAME_BUDGET
		prev = np.clip(prev, 0, 1)

		#agent = int(agent.split("_")[-1])

		# Current agent's values
		agent_means = normalized_means[agent]
		agent_budget = normalized_budgets[agent]
		agent_forwards_needed = normalized_forwards_needed[agent]
		agent_defense_needed = normalized_defense_needed[agent]
		agent_goalies_needed = normalized_goalies_needed[agent]
		#agent_prev_bid = prev[agent]

		# Remaining players (excluding agent)
		other_means = np.delete(normalized_means, agent)
		other_budgets = np.delete(normalized_budgets, agent)
		other_forwards_needed = np.delete(normalized_forwards_needed, agent)
		other_defense_needed = np.delete(normalized_defense_needed, agent)
		other_goalies_needed = np.delete(normalized_goalies_needed, agent)
		#other_prev_bids = np.delete(prev, agent)

		obs = np.concatenate([
			[self.GAME_BUDGET / self.GAME_BUDGET],  # Reference scale
			[min_bid],
			[max_bid],
			[agent_budget],  # Current bidder's budget
			[agent_means],  # Current bidder's mean
			[agent_forwards_needed],
			[agent_defense_needed],
			[agent_goalies_needed],
			#[agent_prev_bid],
			other_means,
			other_budgets,
			other_forwards_needed,
			other_defense_needed,
			other_goalies_needed,
			#other_prev_bids,
			[normalized_forwards_left],
			[normalized_defense_left],
			[normalized_goalies_left],
			[nominated_player_mean],
			nominated_pos_one_hot,
			forwards,
			defensemen,
			goalies
		])

		obs = obs.astype(np.float32)

		return obs


	def get_reward(self):
		rewards = np.zeros(self.num_players)
		MAX_PLAYER_VALUE = 250
		POSITION_WEIGHTS = {
			self.Position.FORWARD: 0.7,
			self.Position.DEFENSEMAN: 0.9,
			self.Position.GOALIE: 1.2
		}

		if self.game_over:
			# Final score components


			# Budget utilization penalty (lose points for leftover budget)
			budget_left_ratios = self.budgets / self.GAME_BUDGET
			budget_penalty = budget_left_ratios * 0.5  # Max 0.5 penalty for full budget

			final_scores = (self.members_means) # - budget_penalty
			mean_score = np.mean(final_scores)

			# Ranking reward [-1 to 1]
			#rankings = np.argsort(final_scores)[::-1]
			#ranking_reward = np.linspace(1.0, -1.0, num=self.num_players)
			#rewards += ranking_reward[rankings]

			# Relative performance bonus
			rewards += (final_scores - mean_score)/100
			rewards += (rewards == np.max(rewards)).astype(np.float32)

		else:
			#pdb.set_trace()
			# Immediate bidding rewards

			#rewards += self.illegal_bid_penalties / 10

			if self.last_winner != -1 and 1 == 0: #disabling
				MAX_MEAN = 250  # Should match observation normalization
				MAX_PRICE = self.GAME_BUDGET
				HISTORY_COEF = 1/100
				curr_athlete = self.nominated_player
				winner = self.last_winner
				curr_pos = curr_athlete.position
				curr_mean = curr_athlete.mean
				curr_price = self.last_price

				adj = np.zeros(self.num_players)

				#print("old bids:", self.prev_bids)

				curr_bids = np.array(self.prev_bids)

				eps = 1e-6
				scale_steal = 1           # How strongly to reward a steal
				tolerance = 0.02 * MAX_MEAN #like within 5 points?


				# Compare against all previous acquisitions in same position
				for athlete, data in self.history.items():
					# Skip current athlete and other positions/players
					if athlete == curr_athlete:
						continue
					if athlete.position == curr_pos:
						prev_mean = athlete.mean
						prev_price = data["price"]
						prev_bids = np.array(data["bids"])
						prev_winner = data["winner"]
						penalties = np.zeros_like(prev_bids)
						### Penalizing bidding more than previously
						if prev_mean > curr_mean:
							#binary penalty for bidding more
							penalties += np.clip(curr_bids - prev_bids, 0, 100)/100

						adj -= penalties

						quality_diff = np.abs(curr_mean - prev_mean)
						if quality_diff < tolerance:
							steal_reward = np.clip(prev_price - curr_price, 0, 100)/100
							adj[winner]  += steal_reward
							adj[prev_winner] -= steal_reward

						#Reward for winning an essentially-equivalent player for cheaper than previous
				# Apply scaled history-based adjustment
				history_effect = adj * HISTORY_COEF

				history_effect = np.clip(history_effect, -1, 1)
				rewards += history_effect
				#rewards[winner] += history_effect

		# Only keep invalid bid penalty
		'''
		for i in range(self.num_players):
			if self.prev_bids[i] > self.budgets[i]:
				rewards[i] = max(-1.0, rewards[i] - 0.7)
		'''

		return rewards #np.clip(rewards, -1.0, 1.0)

		with open("log.txt", "w") as f:
			f.write(f"Means: {self.members_means}\n")
			f.write(f"Budgets: {self.budgets}\n")
			f.write(f"Forwards Needed: {self.members_forwards_needed}\n")
			f.write(f"Defense Needed: {self.member_defense_needed}\n")
			f.write(f"Goalies Needed: {self.member_goalies_needed}\n")
			f.write(f"Rewards: {rewards}")

		# Final clipping for safety


