import torch
import math

def prepare_batch_state(batch_obs, 
                        pool_seq_length_forward=None, 
                        pool_seq_length_defense=None, 
                        pool_seq_length_goalie=None,
                        device="cpu"):
    """
    Given a batch (list) of game state dictionaries (each containing keys such as:
    
      "NUM_PLAYERS", "GAME_BUDGET", "FORWARDS_NEEDED", "DEFENSEMEN_NEEDED", "GOALIES_NEEDED",
      "members_means", "budgets", "members_forwards_needed", "member_defense_needed", "member_goalies_needed",
      "forwards_left", "defense_left", "goalies_left",
      "nominated_player": {"mean": ..., "position": ...},
      "current_bid",
      "athletes_left": {"forward": [...], "defenseman": [...], "goalie": [...]},
      
      And also the gamma parameters:
         "FORWARDS_SHAPE", "FORWARDS_SCLAE",
         "DEFENSE_SHAPE", "DEFENSE_SCALE",
         "GOALIES_SHAPE", "GOALIES_SCLAE"
         
    )
    
    This helper builds the inputs for the network by extracting and normalizing only the needed fields.
    
    The returned dictionary contains:
      - "global":    Tensor of shape (B, 8)
          [normalized forwards_left, defense_left, goalies_left, current_bid,
           normalized nominated_player_mean, nominated_player_position_one_hot (3 dims)]
      - "our":       Tensor of shape (B, 5) for player 0, with features:
           [our_forwards_needed, our_defense_needed, our_goalies_needed, our_budget, our_cumulative_value]
      - "opponents": Tensor of shape (B, NUM_PLAYERS-1, 5) where each opponent is represented by:
           [budget, cumulative_value, forwards_needed, defense_needed, goalies_needed]
         (The opponents are sorted by budget, highest first.)
      - "pool_forward": Tensor of shape (B, pool_seq_length_forward) built from normalized athlete means for forwards.
      - "pool_defense": Tensor of shape (B, pool_seq_length_defense) built from normalized athlete means for defensemen.
      - "pool_goalie":  Tensor of shape (B, pool_seq_length_goalie) built from normalized athlete means for goalies.
    
    Domain normalization details:
      - Roster counts are divided by the maximum possible (num_players * position_needed).
      - current_bid is divided by GAME_BUDGET.
      - Athlete means (both for nominated and pool athletes) are normalized as:
            (x - (shape * scale)) / (scale * sqrt(shape))
        using the gamma parameters for that position.
      - Budgets are divided by GAME_BUDGET.
      - Cumulative team value is divided by an expected team value computed as:
            sum( number_needed * (shape * scale) )  across positions.
    
    If a pool_seq_length is not provided, it defaults to NUM_PLAYERS * (position_needed) for that position.
    """
    
    global_list    = []   # will be list of lists (each length 8)
    our_list       = []   # list of 5-dim vectors (for player 0)
    opponents_list = []   # list (per state) of (NUM_PLAYERS-1) x 5 vectors
    pool_forward_list = []
    pool_defense_list = []
    pool_goalie_list  = []
    
    for state in batch_obs:
        # Basic game parameters:
        num_players    = state["NUM_PLAYERS"]
        game_budget    = state["GAME_BUDGET"]
        forwards_needed = state["FORWARDS_NEEDED"]
        defense_needed  = state["DEFENSEMEN_NEEDED"]
        goalies_needed  = state["GOALIES_NEEDED"]

        player_id = state["PLAYER_ID"]
        
        # Gamma parameters (note the keys "SCLAE" for forwards/goalies):
        f_shape = state["FORWARDS_SHAPE"]
        f_scale = state["FORWARDS_SCALE"]
        d_shape = state["DEFENSE_SHAPE"]
        d_scale = state["DEFENSE_SCALE"]
        g_shape = state["GOALIES_SHAPE"]
        g_scale = state["GOALIES_SCALE"]
        
        # Expected team value for normalization of cumulative values.
        expected_team_value = (
            forwards_needed * f_shape * f_scale +
            defense_needed * d_shape * d_scale +
            goalies_needed * g_shape * g_scale
        )
        
        # ---------------------------------
        # Build the "global" branch vector.
        # 7 entries:
        #   3: normalized roster counts left (forwards, defense, goalies)
        #   1: normalized nominated_player.mean
        #   3: one-hot encoding for nominated_player.position (order: forward, defenseman, goalie)
        gl = []
        gl.append(state["forwards_left"] / (num_players * forwards_needed))
        gl.append(state["defense_left"] / (num_players * defense_needed))
        gl.append(state["goalies_left"] / (num_players * goalies_needed))
        
        nominated_mean = state["nominated_player"]["mean"]
        nominated_position = state["nominated_player"]["position"].lower()  # ensure lowercase
        
        if nominated_position == "forward":
            shape = f_shape
            scale = f_scale
        elif nominated_position == "defenseman":
            shape = d_shape
            scale = d_scale
        elif nominated_position == "goalie":
            shape = g_shape
            scale = g_scale
        else:
            # Fallback: use forward parameters
            shape = f_shape
            scale = f_scale
            
        norm_nominated_mean = (nominated_mean - (shape * scale)) / (scale * math.sqrt(shape))
        gl.append(norm_nominated_mean)
        
        # One-hot encode nominated player's position in order [forward, defenseman, goalie].
        if nominated_position == "forward":
            pos_one_hot = [1.0, 0.0, 0.0]
        elif nominated_position == "defenseman":
            pos_one_hot = [0.0, 1.0, 0.0]
        elif nominated_position == "goalie":
            pos_one_hot = [0.0, 0.0, 1.0]
        else:
            pos_one_hot = [0.0, 0.0, 0.0]
        gl.extend(pos_one_hot)
        global_list.append(gl)
        
        # ---------------------------------
        # Build the "our" branch vector (for player 0)
        our_forwards_needed = state["members_forwards_needed"][player_id] / forwards_needed
        our_defense_needed  = state["member_defense_needed"][player_id] / defense_needed
        our_goalies_needed  = state["member_goalies_needed"][player_id] / goalies_needed
        our_budget          = state["budgets"][player_id] / game_budget
        our_cumulative_value= state["members_means"][player_id] / expected_team_value
        our_vec = [
            our_forwards_needed,
            our_defense_needed,
            our_goalies_needed,
            our_budget,
            our_cumulative_value
        ]
        our_list.append(our_vec)
        
        # ---------------------------------
        # Build the "opponents" branch.
        # For each opponent (players 1 ... num_players-1), include:
        #   [budget, cumulative_value, forwards_needed, defense_needed, goalies_needed]
        # and sort the opponents by budget (largest first).
        opps = []
        opponent_indices = [x for x in range(num_players) if x != player_id]
        # Sort indices by budget (descending)
        opponent_indices = sorted(opponent_indices, key=lambda i: state["budgets"][i], reverse=True)
        for i in opponent_indices:
            opp_budget = state["budgets"][i] / game_budget
            opp_cumulative_value = state["members_means"][i] / expected_team_value
            opp_forwards_needed = state["members_forwards_needed"][i] / forwards_needed
            opp_defense_needed  = state["member_defense_needed"][i] / defense_needed
            opp_goalies_needed  = state["member_goalies_needed"][i] / goalies_needed
            opp_vec = [
                opp_budget,
                opp_cumulative_value,
                opp_forwards_needed,
                opp_defense_needed,
                opp_goalies_needed
            ]
            opps.append(opp_vec)
        opponents_list.append(opps)
        
        # ---------------------------------
        # Build the pool branches.
        # For each position, sort the athlete means (largest first), normalize and pad/truncate.

        #### Shifting this so everything is in terms of percent different from nominated mean
        
        # For forwards:
        pool_fwd_raw = sorted(state["athletes_left"]["forward"], reverse=True)
        target_len_fwd = pool_seq_length_forward if pool_seq_length_forward is not None else num_players * forwards_needed
        norm_pool_fwd = []
        for x in pool_fwd_raw:
            #norm_x = (x - (f_shape * f_scale)) / (f_scale * math.sqrt(f_shape))
            norm_x = (x - nominated_mean)/nominated_mean
            norm_pool_fwd.append(norm_x)
        if len(norm_pool_fwd) < target_len_fwd:
            norm_pool_fwd.extend([-1]*(target_len_fwd - len(norm_pool_fwd)))
        else:
            norm_pool_fwd = norm_pool_fwd[:target_len_fwd]
        pool_forward_list.append(norm_pool_fwd)
        
        # For defensemen:
        pool_def_raw = sorted(state["athletes_left"]["defenseman"], reverse=True)
        target_len_def = pool_seq_length_defense if pool_seq_length_defense is not None else num_players * defense_needed
        norm_pool_def = []
        for x in pool_def_raw:
            #norm_x = (x - (d_shape * d_scale)) / (d_scale * math.sqrt(d_shape))
            norm_x = (x - nominated_mean)/nominated_mean
            norm_pool_def.append(norm_x)
        if len(norm_pool_def) < target_len_def:
            norm_pool_def.extend([-1]*(target_len_def - len(norm_pool_def)))
        else:
            norm_pool_def = norm_pool_def[:target_len_def]
        pool_defense_list.append(norm_pool_def)
        
        # For goalies:
        pool_goal_raw = sorted(state["athletes_left"]["goalie"], reverse=True)
        target_len_goal = pool_seq_length_goalie if pool_seq_length_goalie is not None else num_players * goalies_needed
        norm_pool_goal = []
        for x in pool_goal_raw:
            #norm_x = (x - (g_shape * g_scale)) / (g_scale * math.sqrt(g_shape))
            norm_x = (x - nominated_mean)/nominated_mean
            norm_pool_goal.append(norm_x)
        if len(norm_pool_goal) < target_len_goal:
            norm_pool_goal.extend([-1]*(target_len_goal - len(norm_pool_goal)))
        else:
            norm_pool_goal = norm_pool_goal[:target_len_goal]
        pool_goalie_list.append(norm_pool_goal)
    
    # Convert lists to torch tensors.
    global_tensor    = torch.tensor(global_list, dtype=torch.float32, device=device)     # shape: (B, 8)
    our_tensor       = torch.tensor(our_list, dtype=torch.float32, device=device)        # shape: (B, 5)
    opponents_tensor = torch.tensor(opponents_list, dtype=torch.float32, device=device)  # shape: (B, NUM_PLAYERS-1, 5)
    pool_forward_tensor = torch.tensor(pool_forward_list, dtype=torch.float32, device=device)  # shape: (B, target_len_fwd)
    pool_defense_tensor = torch.tensor(pool_defense_list, dtype=torch.float32, device=device)  # shape: (B, target_len_def)
    pool_goalie_tensor  = torch.tensor(pool_goalie_list, dtype=torch.float32, device=device)   # shape: (B, target_len_goal)
    
    return {
        "global": global_tensor,
        "our": our_tensor,
        "opponents": opponents_tensor,
        "pool_forward": pool_forward_tensor,
        "pool_defense": pool_defense_tensor,
        "pool_goalie": pool_goalie_tensor
    }
