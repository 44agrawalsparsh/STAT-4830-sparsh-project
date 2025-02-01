import json
import sys
from auction_env import AuctionEnv

def load_history(file_path):
    with open(file_path, 'r') as file:
        data = file.read()
    return eval(data, {"np": __import__("numpy"), "Athlete": AuctionEnv.Athlete, "Position": AuctionEnv.Position})

def format_roster(history, strategies_json, player_scores):
    teams = {i: [] for i in range(6)}
    for athlete, details in history.items():
        teams[details['winner']].append({
            'mean': round(athlete.mean, 2),
            'position': athlete.position,
            'price': round(details['price'], 2),
            'strategy': strategies_json[details['winner']],
            'score': player_scores[str(details['winner'])]
        })
    return teams

def print_roster(teams, file=sys.stdout):
    for team, roster in teams.items():
        total_mean = sum(player['mean'] for player in roster)
        total_score = sum(player['score'] for player in roster)
        print(f"Team {team} (Strategy: {roster[0]['strategy']}):", file=file)
        print(f"{'Mean':<10}{'Position':<15}{'Price':<5}{'Score':<5}", file=file)
        for player in sorted(roster, key=lambda x: x['mean']):
            print(f"{player['mean']:<10.2f} {player['position']:<15} {player['price']:<5.2f} {player['score']:<5.2f}", file=file)
        print(f"{'Total':<10} {total_mean:<10.2f} {total_score:<5.2f}", file=file)
        print("\n", file=file)

def main(history_file_path, strategies_json_path, output_file_path):
    history = load_history(history_file_path)
    with open(strategies_json_path, 'r') as file:
        strategies_json = json.load(file)
    with open(history_file_path, 'r') as file:
        lines = file.readlines()
        history = eval(lines[0], {"np": __import__("numpy"), "Athlete": AuctionEnv.Athlete, "Position": AuctionEnv.Position})
        player_scores = json.loads(lines[1])
    teams = format_roster(history, strategies_json, player_scores)
    with open(output_file_path, 'w') as file:
        print_roster(teams, file)

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python format_roster.py <history_file_path> <strategies_json_path> <output_file_path>")
        sys.exit(1)
    
    history_file_path = sys.argv[1]
    strategies_json_path = sys.argv[2]
    output_file_path = sys.argv[3]
    main(history_file_path, strategies_json_path, output_file_path)


