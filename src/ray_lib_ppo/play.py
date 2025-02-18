import os
import ray
import torch
import numpy as np
import json
from ray.rllib.algorithms.ppo import PPO
from auction_env import AuctionEnv
from ray.tune.registry import register_env  # Import register_env

def convert_numpy(obj):
    """Convert NumPy objects to native Python types."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.int32, np.int64, np.float32, np.float64)):
        return obj.item()
    else:
        return str(obj)

def env_creator(env_config):
    """
    Creates a new instance of AuctionEnv.
    The env_config dictionary can override default parameters.
    """
    return AuctionEnv(
        num_players=env_config.get("num_players", 2),
        budget=env_config.get("budget", 1000),
        num_forwards=env_config.get("num_forwards", 12),
        num_defensemen=env_config.get("num_defensemen", 6),
        num_goalies=env_config.get("num_goalies", 2)
    )

# Register your custom environment so RLlib can find it.
register_env("AuctionEnv-v0", lambda config: env_creator(config))
GAME_BUDGET = 1000
# Create a temporary instance to extract observation and action spaces.
temp_env = AuctionEnv(num_forwards=12, num_defensemen=6, num_goalies=2)
per_agent_obs_space = temp_env.observation_space["agent_0"]
per_agent_act_space = temp_env.action_space["agent_0"]

# Define a configuration that matches your training setup.
config = {
    "_enable_new_api_stack": False,
    "enable_connectors": False,
    "env": "AuctionEnv-v0",  # Registered env name.
    "env_config": {
        "num_players": 2,
        "budget": GAME_BUDGET,
        "num_forwards": 12,
        "num_defensemen": 6,
        "num_goalies": 2,
    },
    "framework": "torch",
    "torch_device": "mps",  # Change to "cpu" or "cuda" if needed.
    # Minimal multiagent setup for consistency.
    "multiagent": {
        "policies": {
            "current_policy": (None, per_agent_obs_space, per_agent_act_space, {}),
        },
        "policy_mapping_fn": lambda agent_id, episode, **kwargs: "current_policy",
        "policies_to_train": ["current_policy"],
    }
}

def perform_inference_from_saved_model(saved_algorithm, obs):
    """
    Given a trained algorithm and observation, perform inference using the policy's model.
    """
    actions = {}
    for agent_id, agent_obs in obs.items():
        policy_id = "current_policy"
        rl_module = saved_algorithm.get_module(policy_id)
        # Prepare the observation as a batch (size=1).
        fwd_ins = {"obs": torch.tensor(agent_obs, dtype=torch.float32).unsqueeze(0)}
        fwd_outputs = rl_module.forward_inference(fwd_ins)
        action_dist_class = rl_module.get_inference_action_dist_cls()
        action_dist = action_dist_class.from_logits(fwd_outputs["action_dist_inputs"])
        action = action_dist.sample().numpy()
        # For Box(0,1) actions, apply a sigmoid.
        action = 1 / (1 + np.exp(-action))
        actions[agent_id] = action
    return actions

def human_vs_model(model_checkpoint):
    """
    Loads the model from the checkpoint and plays one game.
    Assumes a two-agent game where you control "agent_0".
    """
    model = PPO(config=config)
    model.restore_from_path(model_checkpoint)
    env = env_creator(config["env_config"])
    obs, info = env.reset()
    done = {"__all__": False}



    print("\n=== Human vs. Model ===")
    print("You are controlling 'agent_0'. Enter your action when prompted.")

    while not done["__all__"]:
        actions = {}
        for agent_id, agent_obs in obs.items():
            if agent_id == "agent_0":
                # Show the human-friendly observation.
                human_obs = env.get_human_obs(agent_id)
                print(f"\nObservation for {agent_id}:")
                print(human_obs)

                idx = int(agent_id.split("_")[1])
                while True:
                    try:
                        bid_input = input("Enter your bid: ")
                        action = int(bid_input) + 0.5
                        # Normalize by the current budget.
                        action /= GAME_BUDGET
                        action = max(0, min(action, 1))
                        break
                    except ValueError:
                        print("Invalid input. Please enter a valid integer bid.")
                actions[agent_id] = action
            else:
                # For the other agent(s), use the trained model.
                actions[agent_id] = perform_inference_from_saved_model(model, {agent_id: agent_obs})[agent_id]
        obs, rewards, done, trunc, info = env.step(actions)
        print("Rewards:", rewards)

    print("Game Over!")
    print(env.get_game_log())

def main():
    # Read the best model checkpoint path from best_model_name.txt.
    checkpoint_file = "best_model_name.txt"
    if not os.path.exists(checkpoint_file):
        print("Error: best_model_name.txt not found.")
        return

    with open(checkpoint_file, "r") as f:
        checkpoint_path = f.read().strip()

    if not checkpoint_path:
        raise ValueError("Checkpoint path is empty.")

    print("Loading model from:", checkpoint_path)

    ray.init(ignore_reinit_error=True)
    try:
        while True:
            human_vs_model(checkpoint_path)
            play_again = input("\nPlay another game? (y/n): ")
            if play_again.strip().lower() not in ["y", "yes"]:
                break
    finally:
        ray.shutdown()

if __name__ == "__main__":
    main()
