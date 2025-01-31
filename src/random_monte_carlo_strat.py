import numpy as np
import copy
import time
import json
import sys
from multiprocessing import Pool
from open_ai_gym_env import AuctionGymEnv
from auction_env import AuctionEnv

def monte_carlo_simulation(env, action, num_simulations=10):
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
    return monte_carlo_simulation(env, action, num_simulations)

def monte_carlo_strategy(env, num_simulations=100):
    best_action = None
    best_score = -np.inf
    num_processes = 4
    simulations_per_process = num_simulations // num_processes

    with Pool(num_processes) as pool:
        for action in range(env.action_space.n):
            scores = pool.map(monte_carlo_worker, [(env, action, simulations_per_process) for _ in range(num_processes)])
            score = np.mean(scores)
            action_text = "pass" if action == 0 else "bid"
            print(f"Action: {action_text}, Score: {score}")
            if score > best_score:
                best_score = score
                best_action = action
    return best_action

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python random_monte_carlo_strat.py <export_file>")
        sys.exit(1)

    export_file = sys.argv[1]
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
    while not done:
        print(f"{env.env.current_bidder} is deciding whether to pass or bid. Current bid is {env.env.current_bid} for {env.env.nominated_player}")
        print(f"This player is ranked {athlete_rankings[env.env.nominated_player.rand]}")
        action = monte_carlo_strategy(env)
        action_text = "pass" if action == 0 else "bid"
        print(f"Decision: {action_text}")
        state, reward, done, info = env.step(action)
        env.render()

    with open(export_file, 'w') as f:
        f.write(str(env.env.history))

    env.close()
