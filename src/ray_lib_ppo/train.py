import ray
from ray import tune
from ray.tune.logger import pretty_print
from ray.rllib.algorithms.ppo import PPO
from ray.tune.registry import register_env
from auction_env import AuctionEnv
from ray.rllib.policy.policy import Policy
import torch
import numpy as np
import os
import random
import json

base_config = {
                    "num_players": 2,
                    "budget": 1000,
                    "num_forwards": 12,
                    "num_defensemen": 6,
                    "num_goalies": 2,
                }


def reverse_engineer_observation(obs, base_config):
    """Reverse engineers the normalized observation vector into a dictionary of unnormalized values.

    Args:
        obs: The normalized observation vector (NumPy array).
        base_config: A dictionary containing the base configuration parameters.

    Returns:
        A dictionary containing the unnormalized observation values.
    """

    num_players = base_config["num_players"]
    GAME_BUDGET = base_config["budget"]
    FORWARDS_NEEDED = base_config["num_forwards"]
    DEFENSEMEN_NEEDED = base_config["num_defensemen"]
    GOALIES_NEEDED = base_config["num_goalies"]

    max_player_value = 400  # Consistent with the original normalization
    max_team_size = FORWARDS_NEEDED + DEFENSEMEN_NEEDED + GOALIES_NEEDED
    max_forwards = FORWARDS_NEEDED * num_players
    max_defense = DEFENSEMEN_NEEDED * num_players
    max_goalies = GOALIES_NEEDED * num_players

    obs_dict = {}

    idx = 0

    obs_dict["game_budget"] = obs[idx] * GAME_BUDGET  # Should be 1 * GAME_BUDGET
    idx += 1
    obs_dict["min_bid"] = obs[idx]  # No unnormalization needed as it was normalized to 1
    idx += 1
    obs_dict["max_bid"] = obs[idx] * GAME_BUDGET # Unnormalize max bid
    idx += 1
    obs_dict["agent_budget"] = obs[idx] * GAME_BUDGET
    idx += 1
    obs_dict["agent_mean"] = obs[idx] * max_player_value
    idx += 1
    obs_dict["agent_forwards_needed"] = obs[idx] * FORWARDS_NEEDED
    idx += 1
    obs_dict["agent_defense_needed"] = obs[idx] * DEFENSEMEN_NEEDED
    idx += 1
    obs_dict["agent_goalies_needed"] = obs[idx] * GOALIES_NEEDED
    idx += 1
    #obs_dict["agent_prev_bid"] = obs[idx] * GAME_BUDGET
    #idx += 1

    obs_dict["other_means"] = obs[idx:idx + num_players - 1] * max_player_value
    idx += num_players - 1
    obs_dict["other_budgets"] = obs[idx:idx + num_players - 1] * GAME_BUDGET
    idx += num_players - 1
    obs_dict["other_forwards_needed"] = obs[idx:idx + num_players - 1] * FORWARDS_NEEDED
    idx += num_players - 1
    obs_dict["other_defense_needed"] = obs[idx:idx + num_players - 1] * DEFENSEMEN_NEEDED
    idx += num_players - 1
    obs_dict["other_goalies_needed"] = obs[idx:idx + num_players - 1] * GOALIES_NEEDED
    idx += num_players - 1
    #obs_dict["other_prev_bids"] = obs[idx:idx + num_players - 1] * GAME_BUDGET
    #idx += num_players - 1

    obs_dict["forwards_left"] = obs[idx] * max_forwards
    idx += 1
    obs_dict["defense_left"] = obs[idx] * max_defense
    idx += 1
    obs_dict["goalies_left"] = obs[idx] * max_goalies
    idx += 1
    obs_dict["nominated_player_mean"] = obs[idx] * max_player_value
    idx += 1

    nominated_pos_one_hot = obs[idx:idx + 3]
    idx += 3

    pos_map = {0: "FORWARD", 1: "DEFENSEMAN", 2: "GOALIE"}
    nominated_pos = "UNKNOWN"
    for i, val in enumerate(nominated_pos_one_hot):
        if val == 1:
            nominated_pos = pos_map.get(i, "UNKNOWN")
            break

    obs_dict["nominated_player_position"] = nominated_pos

    obs_dict["forwards"] = obs[idx:idx + max_forwards] * max_player_value
    idx += max_forwards
    obs_dict["defensemen"] = obs[idx:idx + max_defense] * max_player_value
    idx += max_defense
    obs_dict["goalies"] = obs[idx:idx + max_goalies] * max_player_value
    idx += max_goalies

    return obs_dict

###############################
# Some heuristic policies
###############################
class RandomPolicy(Policy):
    """Random bid between 0 and remaining budget"""
    def compute_actions(self,
                        obs_batch,
                        state_batches=None,
                        prev_action_batch=None,
                        prev_reward_batch=None,
                        info_batch=None,
                        episodes=None,
                        **kwargs):
        # Assuming obs contains [normalized_budget, other_features...]
        budgets = np.array([obs[0] for obs in obs_batch]) * 1000  # De-normalize
        actions = np.random.uniform(0, 1, size=len(obs_batch))  # Random normalized bid
        return actions, [], {}

class ConservativePolicy(Policy):
    """Never bids more than 20% of remaining budget"""
    def compute_actions(self,
                        obs_batch,
                        state_batches=None,
                        prev_action_batch=None,
                        prev_reward_batch=None,
                        info_batch=None,
                        episodes=None,
                        **kwargs):
        actions = []
        for obs in obs_batch:
            budget = obs[0] * 1000  # De-normalize budget
            max_bid = budget * 0.2
            action = np.random.uniform(0, max_bid/1000)  # Normalized
            actions.append(action)
        return np.array(actions), [], []

class AggressivePolicy(Policy):
    """Bids 70-100% of remaining budget"""
    def compute_actions(self,
                        obs_batch,
                        state_batches=None,
                        prev_action_batch=None,
                        prev_reward_batch=None,
                        info_batch=None,
                        episodes=None,
                        **kwargs):
        actions = []
        for obs in obs_batch:
            budget = obs[0] * 1000
            action = np.random.uniform(0.7, 1.0)  # 70-100% of budget
            actions.append(action)
        return np.array(actions), [], []

class ProportionalExcessPolicy(Policy):
    def compute_actions(self,
                        obs_batch,
                        state_batches=None,
                        prev_action_batch=None,
                        prev_reward_batch=None,
                        info_batch=None,
                        episodes=None,
                        **kwargs):
        actions = []
        for obs in obs_batch:
            obs = reverse_engineer_observation(obs)
            obs["forwards"] = [x for x in obs["forwards"] if x > 0]
            min_f = min(obs["forwards"])
            obs["forwards"] = [x - min_f for x in obs["forwards"]]
            obs["defensemen"] = [x for x in obs["defensemen"] if x > 0]
            min_d = min(obs["defensemen"])
            obs["defensemen"] = [x - min_d for x in obs["defensemen"]]
            obs["goalies"] = [x for x in obs["goalies"] if x > 0]
            min_g = min(obs["goalies"])
            obs["goalies"] = [x - min_g for x in obs["goalies"]]

            if obs["nominated_player_position"] == "FORWARD":
                obs["nominated_player_mean"] -= min_f
            elif obs["nominated_player_position"] == "DEFENSEMAN":
                obs["nominated_player_mean"] -= min_d
            elif obs["nominated_player_position"] == "GOALIE":
                obs["nominated_player_mean"] -= min_g

            total_val = sum(obs["forwards"]) + sum(obs["defensemen"]) + sum(obs["goalies"]) + obs["nominated_player_mean"]
            prop = obs["nominated_player_mean"]/total_val

            total_budget = sum(obs["other_budgets"]) + obs["agent_budget"]

            return int(prop * total_budget + 0.5)
        return np.array(actions), [], []

# =============================================
# Update Configuration
# =============================================
heuristic_policies = ["random_policy", "conservative_policy", "aggressive_policy", "proportional", "proportional", "proportional", "proportional", "proportional"]

opponent_pool = []

def convert_numpy(obj):
    """Convert NumPy objects to native Python types."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()  # Convert arrays to lists
    elif isinstance(obj, (np.int32, np.int64, np.float32, np.float64)):
        return obj.item()  # Convert scalars to Python int/float
    else:
        return str(obj)

##############################################
# Environment & Policy Setup
##############################################
def env_creator(config):
    """
    Creates a new instance of AuctionEnv.
    The config can be used to override default parameters.
    """
    return AuctionEnv(
        num_players=config.get("num_players", 2),
        budget=config.get("budget", 1000),
        num_forwards=config.get("num_forwards", 10),
        num_defensemen=config.get("num_defensemen", 5),
        num_goalies=config.get("num_goalies", 2)
    )

# Register the environment so that RLlib can find it by name.
register_env("AuctionEnv-v0", env_creator)
'''
def policy_mapping_fn(agent_id, episode, **kwargs):
    """
    Maps each agent id to a policy.
    """
    return "shared_policy"

def policy_mapping_fn(agent_id, episode, **kwargs):
    if agent_id == "agent_0":
        return "current_policy"
    if opponent_pool and random.random() < 0.7:
        return random.choice(opponent_pool)  # Now, opponent_pool should contain policy IDs like "opponent_0"
    return "current_policy"
'''
def policy_mapping_fn(agent_id, episode, **kwargs):
    if agent_id == "agent_0":
        return "current_policy"
    return "random_policy"

    rand = random.random()
    if opponent_pool and rand < 0.2:
        return random.choice(opponent_pool)  # Now, opponent_pool should contain policy IDs like "opponent_0"
    elif rand < 0.6:
        return random.choice(heuristic_policies)
    return "current_policy"


# Create a temporary instance to extract observation and action spaces.
temp_env = AuctionEnv(num_forwards=12, num_defensemen=6, num_goalies=2)
per_agent_obs_space = temp_env.observation_space["agent_0"]
per_agent_act_space = temp_env.action_space["agent_0"]

##############################################
# PPO Configuration
##############################################
# Note: We increased the entropy_coeff a bit for extra exploration.
config = {
    "_enable_new_api_stack": False,
    "enable_connectors": False,
    "env": "AuctionEnv-v0",
    "env_config": {
        "num_players": 2,
        "budget": 1000,
        "num_forwards": 12,
        "num_defensemen": 6,
        "num_goalies": 2,
    },
    "framework": "torch",
    "torch_device": "mps",  # Use MPS for device

    # Training Tweaks
    "rollout_fragment_length": 64,
    "train_batch_size": 2048,  # Must be divisible by sgd_minibatch_size
    "sgd_minibatch_size": 256,
    "num_sgd_iter": 15,
    "lr": [
        [0, 1e-2],
        [1000, 1e-3]
    ],
    # PPO-Specific Tweaks
    "clip_param": 0.1,
    "kl_target": 0.01,
    "vf_loss_coeff": 0.5,
    # Increase entropy_coeff a little bit for more exploration
    "entropy_coeff": [
    [0, 0.02],        # At timestep 0, entropy coeff is 0.007.
    [1000, 0.0001]  # By timestep 1,000,000, it's decayed to 0.0001.
    ],
    "grad_clip": 0.5,

    # Model: No LSTM, Uses Attention
    "model": {
        "fcnet_hiddens": [256, 256],
        "fcnet_activation": "relu",
        "use_attention": True,
        "attention_num_transformer_units": 1,
        "attention_dim": 64,
        "attention_memory_inference": 20,
        "attention_memory_training": 20,
        "attention_num_heads": 4,
        "attention_head_dim": 32,
        # Don't want this - "attention_use_n_prev_actions": 5,  # Helps attention with memory
    },

    # Multi-Agent Setup
    #"multiagent": {
    #    "policies": {
    #        "shared_policy": (
    #            None,
    #            per_agent_obs_space,
    #            per_agent_act_space,
    #            {}
    #        ),
    #    },
    #    "policy_mapping_fn": policy_mapping_fn,
    #},

    "multiagent": {
        "policies": {
            "current_policy": (None, per_agent_obs_space, per_agent_act_space, {}),
            # Heuristic policies
            "random_policy": (RandomPolicy, per_agent_obs_space, per_agent_act_space, {}),
            "conservative_policy": (ConservativePolicy, per_agent_obs_space, per_agent_act_space, {}),
            "aggressive_policy": (AggressivePolicy, per_agent_obs_space, per_agent_act_space, {}),
            "proportional" : (ProportionalExcessPolicy, per_agent_obs_space, per_agent_act_space, {})
        },
        "policy_mapping_fn": policy_mapping_fn,
        "policies_to_train": ["current_policy"],
    }
    # Random Seed for Reproducibility
    #"seed": 42,
}

##############################################
# Inference Function
##############################################
def perform_inference_from_saved_model(saved_algorithm, obs):
    """
    Given a trained algorithm and observation, perform inference using the policy's model.
    """
    actions = {}
    for agent_id, agent_obs in obs.items():
        policy_id = "current_policy"
        rl_module = saved_algorithm.get_module(policy_id)
        # Prepare the observation as a batch of size 1.
        fwd_ins = {"obs": torch.tensor(agent_obs, dtype=torch.float32).unsqueeze(0)}
        fwd_outputs = rl_module.forward_inference(fwd_ins)
        action_dist_class = rl_module.get_inference_action_dist_cls()
        action_dist = action_dist_class.from_logits(fwd_outputs["action_dist_inputs"])
        action = action_dist.sample().numpy()
        # For Box(0,1) actions, apply a sigmoid.
        action = 1 / (1 + np.exp(-action))
        actions[agent_id] = action
    return actions

##############################################
# Evaluation: Head-to-Head Competition
##############################################
def evaluate_against_random(model_1_path, num_games=100):
    """
    Evaluates two saved models competing against each other in the same game.
    The first half of agents use model_1; the rest use model_2.
    Returns the total accumulated rewards for each model.
    """
    model_1 = PPO(config=config)
    model_1.restore(model_1_path)

    model_1_rewards = 0
    model_2_rewards = 0

    for game in range(num_games):
        env = env_creator(config["env_config"])
        obs, info = env.reset()
        done = {"__all__": False}

        # Assign agents: first half to model_1, second half to model_2.
        agent_ids = list(obs.keys())
        half = len(agent_ids) // 2
        model_1_agents = set(agent_ids[:half])
        model_2_agents = set(agent_ids[half:])

        while not done["__all__"]:
            actions = {}
            # Get actions from model_1.
            for agent_id in model_1_agents:
                actions[agent_id] = perform_inference_from_saved_model(model_1, {agent_id: obs[agent_id]})[agent_id]
            # Get actions from model_2.
            for agent_id in model_2_agents:
                actions[agent_id] = random.random() #perform_inference_from_saved_model(model_2, {agent_id: obs[agent_id]})[agent_id]

            obs, rewards, done, trunc, info = env.step(actions)
            # Accumulate rewards.
            for agent_id, reward in rewards.items():
                if agent_id in model_1_agents:
                    model_1_rewards += reward
                else:
                    model_2_rewards += reward

    return model_1_rewards, model_2_rewards

def evaluate_models(model_1_path, model_2_path, num_games=100):
    """
    Evaluates two saved models competing against each other in the same game.
    The first half of agents use model_1; the rest use model_2.
    Returns the total accumulated rewards for each model.
    """
    model_1 = PPO(config=config)
    model_2 = PPO(config=config)
    model_1.restore(model_1_path)
    model_2.restore(model_2_path)

    model_1_rewards = 0
    model_2_rewards = 0

    for game in range(num_games):
        env = env_creator(config["env_config"])
        obs, info = env.reset()
        done = {"__all__": False}

        # Assign agents: first half to model_1, second half to model_2.
        agent_ids = list(obs.keys())
        half = len(agent_ids) // 2
        model_1_agents = set(agent_ids[:half])
        model_2_agents = set(agent_ids[half:])

        while not done["__all__"]:
            actions = {}
            # Get actions from model_1.
            for agent_id in model_1_agents:
                actions[agent_id] = perform_inference_from_saved_model(model_1, {agent_id: obs[agent_id]})[agent_id]
            # Get actions from model_2.
            for agent_id in model_2_agents:
                actions[agent_id] = perform_inference_from_saved_model(model_2, {agent_id: obs[agent_id]})[agent_id]

            obs, rewards, done, trunc, info = env.step(actions)
            # Accumulate rewards.
            for agent_id, reward in rewards.items():
                if agent_id in model_1_agents:
                    model_1_rewards += reward
                else:
                    model_2_rewards += reward

    return model_1_rewards, model_2_rewards

##############################################
# Main Training Loop with LR Scheduler and King-of-the-Hill Tracking
##############################################
if __name__ == "__main__":
    ray.init(ignore_reinit_error=True)
    trainer = PPO(config=config)
    saved_checkpoints = []

    # For the LR scheduler: keep track of the initial learning rate.
    initial_lr = config["lr"]

    # For king-of-the-hill tracking:
    best_model_path = None
    best_model_reward = -float("inf")

    opp_num = 0
    for i in range(501):
        result = trainer.train()
        print(f"=== Iteration: {i} ===")
        print(pretty_print(result))

        '''

        # Save a checkpoint every 10 iterations (for demonstration purposes).
        if i % 1 == 0:
            checkpoint = trainer.save()
            print("Checkpoint saved at", checkpoint)
            # (Optional) Play a demonstration game.
            eval_env = env_creator(config["env_config"])
            obs, info = eval_env.reset()
            done = {"__all__": False}
            saved_algorithm = PPO(config=config)
            saved_algorithm.restore(checkpoint)
            while not done["__all__"]:
                actions = perform_inference_from_saved_model(saved_algorithm, obs)
                print("Actions:", actions)
                obs, rewards, done, trunc, info = eval_env.step(actions)
            os.makedirs("logs", exist_ok=True)
            with open(f"logs/{i}_log.txt", "w") as f:
                f.write(eval_env.get_game_log())
                f.write("\n====================\n")
        '''

        # Save a checkpoint every 50 iterations and track for king-of-the-hill.
        if i % 10 == 0:
            checkpoint = trainer.save()
            print("Checkpoint saved at", checkpoint)
            saved_checkpoints.append(checkpoint)

            if best_model_path is None:
                best_model_path = checkpoint
                print("Setting initial best model to checkpoint:", checkpoint)
            else:
                # Evaluate the new checkpoint against the current best model.
                #new_model_reward, best_model_eval_reward = evaluate_models(checkpoint, best_model_path, num_games=100)
                new_model_reward, random_reward = evaluate_against_random(checkpoint, num_games=100)
                #print(f"Evaluation: New Model Reward: {new_model_reward}, Best Model Reward: {best_model_eval_reward}")


                if new_model_reward > best_model_reward:
                    best_model_reward = new_model_reward
                    best_model_path = checkpoint

                    abs_path = os.path.abspath("models/best_model")
                    uri_path = "file://" + abs_path

                    # Save the checkpoint to the directory.
                    checkpoint_path = trainer.save_to_path(uri_path)
                    print("Model saved to:", checkpoint_path)

                    # Write the checkpoint path to a file for later use.
                    with open("best_model_name.txt", "w") as f:
                        f.write(checkpoint_path)

                    print("New model becomes the king of the hill!")
                    # Instead of storing the checkpoint, store the new policy id (e.g., "opponent_X")
                    opp_num += 1
                    new_opponent_id = f"opponent_{opp_num}"
                    opponent_pool.append(new_opponent_id)
                    if len(opponent_pool) > 10:
                        opponent_pool.pop(0)

                    # Update the multiagent policies (if possible, before trainer initialization)
                    config["multiagent"]["policies"][new_opponent_id] = (
                        None, per_agent_obs_space, per_agent_act_space, {}
                    )
                else:
                    print("King of the hill remains:", best_model_path)
                # (Optional) Log evaluation results.
                with open(f"logs/{i}_log_eval.txt", "w") as f:
                    f.write(f"New Model Reward: {new_model_reward}, Random Model Reward: {random_reward}")

    ray.shutdown()
