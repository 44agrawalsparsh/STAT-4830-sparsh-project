from basic_game.neural_networks.ppo_nn import PPONNetWrapper
from basic_game.basic_auction import BasicAuctionGame
import os


if __name__ == "__main__":
	game = BasicAuctionGame()
	state = game.getInitState()
	while True:
		os.system('cls' if os.name == 'nt' else 'clear')
		print(state["nominated_player"])
		print("\n\n\n\n")
		athletes_sorted = sorted(state["athletes_left"], key=lambda x: (x["position"], -x["mean"]))
		current_position = None
		for athlete in athletes_sorted:
			if athlete["position"] != current_position:
				if current_position is not None:
					print()
				current_position = athlete["position"]
				print(f"{current_position}: {athlete['mean']}", end="")
			else:
				print(", ", end="")
				print(f"{athlete['mean']}", end="")
		print("\n\n\n\n")
		print("Budgets", state["budgets"])
		print("Means", state["members_means"])
		print("Forwards Needed", state["members_forwards_needed"])
		print("Defense Needed", state["member_defense_needed"])
		print("Goalies Needed", state["member_goalies_needed"])
		print("\n\n\n\n")
		def get_valid_bid(player_name, budget):
			while True:
				try:
					bid = (float(input(f"{player_name}, how much did you bid? ")) + 0.5) / budget
					if 0 <= bid <= 1:
						return bid
					else:
						print("Bid must be between your min and max bids. Please try again.")
				except Exception:
					print("Invalid input. Please enter a number.")

		action = get_valid_bid("Rishabh", state["budgets"][0])
		state = game.getNextState(state, action)
		action = get_valid_bid("Sparsh", state["budgets"][1])
		state = game.getNextState(state, action)
		