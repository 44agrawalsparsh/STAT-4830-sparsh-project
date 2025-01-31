import json
import ast
import sys
from auction_env import AuctionEnv

def load_history(file_path):
    with open(file_path, 'r') as file:
        data = file.read()
    return eval(data, {"np": __import__("numpy"), "Athlete": AuctionEnv.Athlete, "Position": AuctionEnv.Position})

def format_roster(history):
    teams = {i: [] for i in range(6)}
    for athlete, details in history.items():
        teams[details['winner']].append({
            'mean': round(athlete.mean, 2),
            'variance': round(athlete.variance, 2),
            'position': athlete.position,
            'price': round(details['price'], 2)
        })
    return teams

def print_roster(teams):
    for team, roster in teams.items():
        total_mean = sum(player['mean'] for player in roster)
        total_variance = sum(player['variance'] for player in roster)
        print(f"Team {team}:")
        print(f"{'Mean':<10}{'Variance':<10}{'Position':<15}{'Price':<5}")
        for player in sorted(roster, key=lambda x: x['mean']):
            print(f"{player['mean']:<10.2f} {player['variance']:<10.2f} {player['position']:<15} {player['price']:<5.2f}")
        print(f"{'Total':<10}{total_mean:<10.2f}{total_variance:<10.2f}")
        print("\n")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python format_roster.py <history_file_path>")
        sys.exit(1)
    
    history_file_path = sys.argv[1]
    history = load_history(history_file_path)
    teams = format_roster(history)
    print_roster(teams)
