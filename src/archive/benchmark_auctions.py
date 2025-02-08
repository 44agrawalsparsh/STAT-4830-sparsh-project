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

    def monte_carlo_simulation(env, action, num_simulations):
        total_reward = 0
        player_idx = env.env.current_bidder
        for _ in range(num_simulations):
            env_copy = copy.deepcopy(env)
            env_copy.step(action)
            while not env_copy.env.game_over:
                random_action = env_copy.action_space.sample()
                env_copy.step(random_action)

            total_reward += env_copy.get_payouts()[player_idx]  # Assuming get_reward method exists
        return total_reward / num_simulations

    def monte_carlo_worker(args):
        env, action, num_simulations = args
        return MONTE_CARLO_STRATEGY.monte_carlo_simulation(env, action, num_simulations)

    def monte_carlo_strategy(self):
        best_action = None
        best_score = -np.inf
        num_processes = 4
        simulations_per_process = self.num_simulations // num_processes

        scores = run_single_core(self.env, num_processes, simulations_per_process) #run_multiprocessing(self.env, num_processes, simulations_per_process) #run_single_core(self.env, num_processes, simulations_per_process)
        for action, score in scores.items():
            action_text = "pass" if action == 0 else "bid"
            #print(f"Action: {action_text}, Score: {score}")
            if score > best_score:
                best_score = score
                best_action = action
        return best_action
    
def run_single_core(env, num_processes, sims):
    scores = {}
    for action in range(env.action_space.n):
        action_scores = [MONTE_CARLO_STRATEGY.monte_carlo_simulation(env, action, sims) for _ in range(num_processes)]
        scores[action] = np.mean(action_scores)
    return scores

def run_multiprocessing(env, num_processes, simulations_per_process):
    scores = {}
    with Pool(num_processes) as pool:
        for action in range(env.action_space.n):
            action_scores = pool.map(MONTE_CARLO_STRATEGY.monte_carlo_worker, [(env, action, simulations_per_process) for _ in range(num_processes)])
            scores[action] = np.mean(action_scores)
    return scores

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
    
class ALWAYS_BID:
    ''' 
        
        Very naive class of strategies where we always bid
        
    '''

    def __init__(self, env):
        self.env = env
    
    def always_bid(self):
        return 1
    
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
            for athlete in state["athletes_left"]["forward"]:
                pos_sum += athlete
        elif cur_player["position"] == AuctionEnv.Position.DEFENSEMAN:
            rel_budget = defenseman_budget
            for athlete in state["athletes_left"]["defenseman"]:
                pos_sum += athlete
        elif cur_player["position"] == AuctionEnv.Position.GOALIE:
            rel_budget = goalie_budget
            for athlete in state["athletes_left"]["goalie"]:
                pos_sum += athlete

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

        for athlete in state["athletes_left"]["forward"]:
            self.min_values["forward"] = min(self.min_values["forward"], athlete)

        for athlete in state["athletes_left"]["defenseman"]:
            self.min_values["defenseman"] = min(self.min_values["defenseman"], athlete)
        
        for athlete in state["athletes_left"]["goalie"]:  
            self.min_values["goalie"] = min(self.min_values["goalie"], athlete)

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
        add_on_sum -= get_replacement_value(AuctionEnv.Position[cur_player["position"].upper()])
        if cur_player["position"] == AuctionEnv.Position.FORWARD:
            for athlete in state["athletes_left"]["forward"]:
                add_on_sum += athlete
                add_on_sum -= get_replacement_value(AuctionEnv.Position.FORWARD)
        elif cur_player["position"] == AuctionEnv.Position.DEFENSEMAN:
            for athlete in state["athletes_left"]["defenseman"]:
                add_on_sum += athlete
                add_on_sum -= get_replacement_value(AuctionEnv.Position.DEFENSEMAN)
        elif cur_player["position"] == AuctionEnv.Position.GOALIE:
            for athlete in state["athletes_left"]["goalie"]:
                add_on_sum += athlete
                add_on_sum -= get_replacement_value(AuctionEnv.Position.GOALIE)

        var = cur_player["mean"]
        var -= get_replacement_value(AuctionEnv.Position[cur_player["position"].upper()])
        fair = var / add_on_sum * total_budget

        if state["current_bid"] < fair - 1:
            return 1
        
        return 0



def run_auction(export_file, strategies_dict):
    env = AuctionGymEnv()
    state = env.reset()
    done = False

    #print("Auctioning off: ")
    positions = {AuctionEnv.Position.FORWARD: [], AuctionEnv.Position.DEFENSEMAN: [], AuctionEnv.Position.GOALIE: []}
    for player in env.env.archive_athletes:
        positions[player.position].append(player)

    '''
    for position in positions:
        positions[position].sort(key=lambda x: x, reverse=True)
        print(position)
        for i,player in enumerate(positions[position]):
            print(f"{player}, Rank: {i+1}")
    '''
    player_strategies = []
    for i in range(6):
        #print(strategies_dict)
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
        elif strategy_type == 'always_bid':
            player_strategies.append(ALWAYS_BID(env))
        else:
            raise ValueError(f"Unknown strategy type {strategy_type} for player {i}")
        
    '''for i in range(10):
        print()
    print(f"Player {env.env.nominated_player} is now up for auction.")
    '''
    while not done:
        current_bidder = env.env.current_bidder
        #print(f"{current_bidder} is deciding whether to pass or bid. Current bid is {env.env.current_bid} for {env.env.nominated_player}")
        #print(f"This player is ranked {athlete_rankings[env.env.nominated_player.rand]}")
        player = env.env.nominated_player
        
        strategy = player_strategies[current_bidder]
        if isinstance(strategy, MONTE_CARLO_STRATEGY):
            #print("MONTE CARLO STRAT")
            action = strategy.monte_carlo_strategy()
        elif isinstance(strategy, FLIP):
            #print("FLIP STRAT")
            action = strategy.flip_strategy()
        elif isinstance(strategy, PROPORTIONAL_STRATEGY):
            #print("PROPORTIONAL STRAT")
            action = strategy.proportional_strategy()
        elif isinstance(strategy, VALUE_ABOVE_REPLACEMENT_STRATEGY):
            #print("VALUE ABOVE REPLACEMENT STRAT")
            action = strategy.value_above_replacement_strategy()
        elif isinstance(strategy, ALWAYS_BID):
            #print("ALWAYS BID STRAT")
            action = strategy.always_bid()
        
        action_text = "pass" if action == 0 else "bid"
        #print(f"Decision: {action_text}")
        state, reward, done, info = env.step(action)

        env.env = AuctionEnv(state = state)

        print(state)

        '''if env.env.nominated_player != player:
            print(f"SOLD: {env.env.history[player]}")
            print(f"Player {env.env.nominated_player} is now up for auction.")
        '''
        #env.render()

    game_score_arr = env.env.get_game_score()
    player_scores = {int(i): int(game_score_arr[i]) for i in range(len(game_score_arr))}

    with open(export_file, 'w') as f:
        f.write(str(env.env.history))
        f.write("\n")
        f.write(json.dumps(player_scores))

    env.close()

def main():
    if len(sys.argv) != 3:
        print("Usage: python random_monte_carlo_strat.py <export_file> <strategies_json>")
        sys.exit(1)

    export_file = sys.argv[1]
    strategies_json = sys.argv[2]

    with open(strategies_json, 'r') as f:
        strategies_dict = json.load(f)

    run_auction(export_file, strategies_dict)

if __name__ == "__main__":
    main()

