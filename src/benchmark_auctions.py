import numpy as np
import copy
import time
import json
import sys
from multiprocessing import Pool
from open_ai_gym_env import AuctionGymEnv
from auction_env import AuctionEnv

class MONTE_CARLO_STRATEGY:
    def __init__(self, env, num_simulations=100):
        self.env = env
        self.num_simulations = num_simulations

    def monte_carlo_simulation(self, action):
        total_reward = 0
        player_idx = self.env.env.current_bidder
        for _ in range(self.num_simulations):
            env_copy = copy.deepcopy(self.env)
            env_copy.step(action)
            while not env_copy.env.game_over:
                random_action = env_copy.action_space.sample()
                env_copy.step(random_action)

            total_reward += env_copy.get_payouts()[player_idx]  # Assuming get_reward method exists
        return total_reward / self.num_simulations

    def monte_carlo_worker(self, args):
        env, action, num_simulations = args
        return self.monte_carlo_simulation(action)

    def monte_carlo_strategy(self):
        best_action = None
        best_score = -np.inf
        num_processes = 4
        simulations_per_process = self.num_simulations // num_processes

        with Pool(num_processes) as pool:
            for action in range(self.env.action_space.n):
                scores = pool.map(self.monte_carlo_worker, [(self.env, action, simulations_per_process) for _ in range(num_processes)])
                score = np.mean(scores)
                action_text = "pass" if action == 0 else "bid"
                print(f"Action: {action_text}, Score: {score}")
                if score > best_score:
                    best_score = score
                    best_action = action
        return best_action
    
class FLIP:
    ''' 
        
        Very naive class of strategies where we set a probability of bidding and pass accordingly and do that
    '''

    def __init__(self, env, p):
        assert 0 < p < 1, "need a valid probability"
        self.p = p
        self.env = env
    
    def flip_strategy(self):
        return np.random.choice([0, 1], p=[1 - self.p, self.p])
    
class PROPORTIONAL_STRATEGY:
    ''' 
        
        Very naive class of strategies where we take the remaining players and budgets - allocate some budget to the player based on the mean and variance of the player and then bid if the current bid is less than the fair value of the player
        
    '''
    
    def __init__(self, env, weight_vector):
        self.env = env
        self.weight_vector = weight_vector

        if len(self.weight_vector) != 3:
            raise ValueError("Weight vector must have exactly 3 elements.")
        self.weight_vector = np.array(self.weight_vector)
        self.weight_vector = self.weight_vector / np.sum(self.weight_vector)

    def proportional_strategy(self):
        
        state = self.env.env.get_state()
        total_budget = np.sum(state['budgets'])

        forward_budget = self.weight_vector[0] * total_budget
        defenseman_budget = self.weight_vector[1] * total_budget
        goalie_budget = self.weight_vector[2] * total_budget

        cur_player = state["nominated_player"]

        rel_budget = 0
        pos_sum = cur_player["mean"]
        if cur_player["position"] == AuctionEnv.Position.FORWARD:
            rel_budget = forward_budget
            for athlete in state["athletes_left"]["forwards"]:
                pos_sum += athlete[0]
        elif cur_player["position"] == AuctionEnv.Position.DEFENSEMAN:
            rel_budget = defenseman_budget
            for athlete in state["athletes_left"]["defensemen"]:
                pos_sum += athlete[0]
        elif cur_player["position"] == AuctionEnv.Position.GOALIE:
            rel_budget = goalie_budget
            for athlete in state["athletes_left"]["goalies"]:
                pos_sum += athlete[0]

        fair = cur_player["mean"] / pos_sum * rel_budget

        if state["current_bid"] < fair - 1:
            return 1
        
        return 0

class VALUE_ABOVE_REPLACEMENT_STRATEGY:
    ''' 

    Basic strategy of going through each position - subtracting the value by the min of said position - and then calling that VAR. Player's fair value is VAR / remaining budgets.
    
    '''

    def __init__(self, env):

        self.env = env

        state = self.env.env.get_state()

        self.min_values = {
            "forward" : 1e10,
            "defenseman" : 1e10,
            "goalie" : 1e10
        }

        for athlete in state["athletes_left"]["forwards"]:
            self.min_values["forward"] = min(self.min_values["forward"], athlete[0])

        for athlete in state["athletes_left"]["defensemen"]:
            self.min_values["defenseman"] = min(self.min_values["defenseman"], athlete[0])
        
        for athlete in state["athletes_left"]["goalies"]:  
            self.min_values["goalie"] = min(self.min_values["goalie"], athlete[0])

        if state["nominated_player"]["position"] == AuctionEnv.Position.FORWARD:
            self.min_values["forward"] = min(self.min_values["forward"], state["nominated_player"]["mean"])
        elif state["nominated_player"]["position"] == AuctionEnv.Position.DEFENSEMAN:
            self.min_values["defenseman"] = min(self.min_values["defenseman"], state["nominated_player"]["mean"])
        elif state["nominated_player"]["position"] == AuctionEnv.Position.GOALIE:
            self.min_values["goalie"] = min(self.min_values["goalie"], state["nominated_player"]["mean"])

    def value_above_replacement_strategy(self):
        
        state = self.env.env.get_state()

        total_budget = np.sum(state['budgets'])

        cur_player = state["nominated_player"]

        def get_replacement_value(position):
            if position == AuctionEnv.Position.FORWARD:
                return self.min_values["forward"]
            elif position == AuctionEnv.Position.DEFENSEMAN:
                return self.min_values["defenseman"]
            elif position == AuctionEnv.Position.GOALIE:
                return self.min_values["goalie"]
            else:
                raise ValueError("Invalid position")
            
        add_on_sum = cur_player["mean"]
        add_on_sum -= get_replacement_value(cur_player["position"])
        if cur_player["position"] == AuctionEnv.Position.FORWARD:
            for athlete in state["athletes_left"]["forwards"]:
                add_on_sum += athlete[0]
                add_on_sum -= get_replacement_value(AuctionEnv.Position.FORWARD)
        elif cur_player["position"] == AuctionEnv.Position.DEFENSEMAN:
            for athlete in state["athletes_left"]["defensemen"]:
                add_on_sum += athlete[0]
                add_on_sum -= get_replacement_value(AuctionEnv.Position.DEFENSEMAN)
        elif cur_player["position"] == AuctionEnv.Position.GOALIE:
            for athlete in state["athletes_left"]["goalies"]:
                add_on_sum += athlete[0]
                add_on_sum -= get_replacement_value(AuctionEnv.Position.GOALIE)

        var = cur_player["mean"]
        var -= get_replacement_value(cur_player["position"])
        fair = var / add_on_sum * total_budget

        if state["current_bid"] < fair - 1:
            return 1
        
        return 0



def run_auction(export_file, strategies_dict):
    env = AuctionGymEnv()
    state = env.reset()
    done = False

    print("Auctioning off: ")
    positions = {AuctionEnv.Position.FORWARD: [], AuctionEnv.Position.DEFENSEMAN: [], AuctionEnv.Position.GOALIE: []}
    for player in env.env.archive_athletes:
        positions[player.position].append(player)

    for position in positions:
        positions[position].sort(key=lambda x: x, reverse=True)

    athlete_rankings = {}
    for position, players in positions.items():
        for rank, player in enumerate(players, start=1):
            athlete_rankings[player.rand] = rank

    for player in sorted(env.env.athletes, key=lambda x: athlete_rankings[x.rand]):
        print(f"Player {player}, Rank: {athlete_rankings[player.rand]})")

    player_strategies = []
    for i in range(6):
        strategy_info = strategies_dict.get(i)
        if strategy_info is None:
            raise ValueError(f"No strategy provided for player {i}")
        
        strategy_type = strategy_info['type']
        if strategy_type == 'monte_carlo':
            player_strategies.append(MONTE_CARLO_STRATEGY(env, strategy_info.get('num_simulations', 100)))
        elif strategy_type == 'flip':
            player_strategies.append(FLIP(env, strategy_info['p']))
        elif strategy_type == 'proportional':
            player_strategies.append(PROPORTIONAL_STRATEGY(env, strategy_info['weight_vector']))
        elif strategy_type == 'value_above_replacement':
            player_strategies.append(VALUE_ABOVE_REPLACEMENT_STRATEGY(env))
        else:
            raise ValueError(f"Unknown strategy type {strategy_type} for player {i}")

    while not done:
        current_bidder = env.env.current_bidder
        print(f"{current_bidder} is deciding whether to pass or bid. Current bid is {env.env.current_bid} for {env.env.nominated_player}")
        print(f"This player is ranked {athlete_rankings[env.env.nominated_player.rand]}")
        
        strategy = player_strategies[current_bidder]
        if isinstance(strategy, MONTE_CARLO_STRATEGY):
            print("MONTE CARLO STRAT")
            action = strategy.monte_carlo_strategy()
        elif isinstance(strategy, FLIP):
            print("FLIP STRAT")
            action = strategy.flip_strategy()
        elif isinstance(strategy, PROPORTIONAL_STRATEGY):
            print("PROPORTIONAL STRAT")
            action = strategy.proportional_strategy()
        elif isinstance(strategy, VALUE_ABOVE_REPLACEMENT_STRATEGY):
            print("VALUE ABOVE REPLACEMENT STRAT")
            action = strategy.value_above_replacement_strategy()
        
        action_text = "pass" if action == 0 else "bid"
        print(f"Decision: {action_text}")
        state, reward, done, info = env.step(action)
        env.render()

    game_score_arr = env.env.get_game_score()
    player_scores = {int(i): game_score_arr[i] for i in range(len(game_score_arr))}

    with open(export_file, 'w') as f:
        f.write(str(env.env.history))
        f.write("\n")
        f.write(json.dumps(player_scores))

    env.close()

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python random_monte_carlo_strat.py <export_file> <strategies_json>")
        sys.exit(1)

    export_file = sys.argv[1]
    strategies_json = sys.argv[2]

    with open(strategies_json, 'r') as f:
        strategies_dict = json.load(f)

    run_auction(export_file, strategies_dict)

