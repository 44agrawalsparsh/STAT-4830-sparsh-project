import torch
import torch.nn as nn
import torch.nn.functional as F

class AuctionEnvNet(nn.Module):
    def __init__(self, game, pool_seq_length_forward=None, pool_seq_length_defense=None, pool_seq_length_goalie=None):
        """
        Build a network whose input sizes depend on the auction game configuration.
        
        The network has several branches:
          - Global branch: encodes overall game features (remaining roster counts, current bid, nominated player info).
          - Our branch: encodes our team’s state (needs, budget, cumulative value).
          - Opponents branch: processes each opponent’s features (budget, cumulative value, roster needs) independently,
              then concatenates them (assuming opponents are sorted by budget).
          - Pool branches: for each position (forward, defenseman, goalie) we process the sorted athlete means
              (zero padded/truncated to a fixed length) with a few 1D convolution layers.
          
        After merging the branches, the network splits into two heads:
          - A policy head (pi) which outputs logits over actions (here, assumed to be 2 actions: bid/pass).
          - A value head (v) which outputs a scalar value.
        
        Parameters:
          game: an instance of your auction game (assumed to have getStructure() method)
          pool_seq_length_*: fixed length for the sorted athlete lists for each position. If None, we use the full length.
        """
        super(AuctionEnvNet, self).__init__()
        
        # Read game configuration
        structure = game.getStructure()
        self.num_players     = structure["NUM_PLAYERS"]
        self.game_budget     = structure["GAME_BUDGET"]
        self.num_forwards    = structure["FORWARDS_NEEDED"]
        self.num_defensemen  = structure["DEFENSEMEN_NEEDED"]
        self.num_goalies     = structure["GOALIES_NEEDED"]

        # --------------------------
        # Global Branch:
        # – Features: [forwards_left, defense_left, goalies_left, current_bid,
        #            nominated_player_mean, nominated_player_position(one-hot 3-d)]
        # Total input dim = 3 + 1 + 1 + 3 = 8.
        global_input_dim = 8
        global_hidden_dim = 64
        self.global_branch = nn.Sequential(
            nn.Linear(global_input_dim, global_hidden_dim),
            nn.ReLU(),
            nn.Linear(global_hidden_dim, 32),
            nn.ReLU()
        )
        
        # --------------------------
        # Our Branch:
        # – Features: [our_forwards_needed, our_defense_needed, our_goalies_needed, our_budget, our_cumulative_value]
        # Total input dim = 3 + 1 + 1 = 5.
        our_input_dim = 5
        our_hidden_dim = 32
        self.our_branch = nn.Sequential(
            nn.Linear(our_input_dim, our_hidden_dim),
            nn.ReLU(),
            nn.Linear(our_hidden_dim, 16),
            nn.ReLU()
        )
        
        # --------------------------
        # Opponents Branch:
        # – For each opponent (players 1 ... num_players-1), features:
        #   [budget, cumulative_value, forwards_needed, defense_needed, goalies_needed] (5 dims)
        opp_input_dim = 5
        opp_hidden_dim = 16  # shared for each opponent
        self.opp_shared = nn.Sequential(
            nn.Linear(opp_input_dim, opp_hidden_dim),
            nn.ReLU()
        )
        # After processing each opponent, we flatten (num_players-1)*opp_hidden_dim.
        # Then pass through an FC layer.
        opp_total_dim = (self.num_players - 1) * opp_hidden_dim
        self.opponents_branch = nn.Sequential(
            nn.Linear(opp_total_dim, 32),
            nn.ReLU()
        )
        
        # --------------------------
        # Pool Branches (for the athletes remaining)
        # We assume that for each position we are given a sorted list of athlete means.
        # If no fixed length is provided, we use the maximum possible size.
        self.pool_seq_length_forward = self.num_players * self.num_forwards #pool_seq_length_forward or (self.num_players * self.num_forwards)
        self.pool_seq_length_defense = self.num_players * self.num_defensemen #pool_seq_length_defense or (self.num_players * self.num_defensemen)
        self.pool_seq_length_goalie  = self.num_players * self.num_goalies #pool_seq_length_goalie  or (self.num_players * self.num_goalies)
        
        # Define a small conv network for each position.
        def make_pool_branch(seq_length):
            # The network expects input shape: (batch, channels=1, seq_length)
            # We use two conv layers and then adaptive pooling to get a fixed output size.
            return nn.Sequential(
                nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.AdaptiveMaxPool1d(output_size=1)  # output shape: (batch, 32, 1)
            )
        
        self.pool_forward = make_pool_branch(self.pool_seq_length_forward)
        self.pool_defense = make_pool_branch(self.pool_seq_length_defense)
        self.pool_goalie  = make_pool_branch(self.pool_seq_length_goalie)
        
        # --------------------------
        # Merge all branches.
        # Global branch output: 32
        # Our branch output: 16
        # Opponents branch output: 32
        # Pool branch outputs: each produces 32, so 3*32 = 96.
        merged_dim = 32 + 16 + 32 + 96
        self.merged_fc = nn.Sequential(
            nn.Linear(merged_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU()
        )
        
        # --------------------------
        # Heads: Policy and Value
        self.policy_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 2)  # assuming two possible moves: bid or pass
        )
        
        self.value_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, self.num_players),  # scalar value
            nn.Softmax(dim=1)
        )
        
    def forward(self, state):
        """
        Expects state as a dictionary with keys that mirror the structure output by AuctionEnv.get_state().
        For example, state might contain:
          - global: a tensor of shape (batch, 8) for the global branch.
          - our: a tensor of shape (batch, 5) for our branch.
          - opponents: a tensor of shape (batch, num_players-1, 5) for opponents.
          - pool_forward: a tensor of shape (batch, pool_seq_length_forward) for forwards.
          - pool_defense: a tensor of shape (batch, pool_seq_length_defense) for defensemen.
          - pool_goalie: a tensor of shape (batch, pool_seq_length_goalie) for goalies.
        (You may need to write a wrapper to extract these from the full game state.)
        """
        # Global branch
        global_input = state["global"]  # shape (B, 8)
        global_feat = self.global_branch(global_input)  # (B, 32)
        
        # Our branch (player 0)
        our_input = state["our"]  # shape (B, 5)
        our_feat = self.our_branch(our_input)  # (B, 16)
        
        # Opponents branch
        # state["opponents"] is assumed to be of shape (B, num_players-1, 5)
        B = state["opponents"].shape[0]
        num_opp = state["opponents"].shape[1]
        # Process each opponent independently with the shared network.
        opp_feat = self.opp_shared(state["opponents"].view(B * num_opp, -1))  # (B*(num_players-1), opp_hidden_dim)
        opp_feat = opp_feat.view(B, num_opp * opp_feat.shape[-1])  # (B, (num_players-1)*opp_hidden_dim)
        opp_feat = self.opponents_branch(opp_feat)  # (B, 32)
        
        # Pool branches for athletes remaining:
        # Each pool branch input is assumed to be a float tensor of shape (B, seq_length).
        # We add a channel dimension to use with conv1d.
        def process_pool(x, pool_module):
            # x shape: (B, seq_length)
            x = x.unsqueeze(1)  # (B, 1, seq_length)
            x = pool_module(x)  # (B, 32, 1)
            return x.squeeze(-1)  # (B, 32)
        
        pool_fwd  = process_pool(state["pool_forward"], self.pool_forward)  # (B, 32)
        pool_def  = process_pool(state["pool_defense"], self.pool_defense)  # (B, 32)
        pool_goal = process_pool(state["pool_goalie"], self.pool_goalie)    # (B, 32)
        pool_feat = torch.cat([pool_fwd, pool_def, pool_goal], dim=-1)  # (B, 96)
        
        # Merge all branch features:
        merged = torch.cat([global_feat, our_feat, opp_feat, pool_feat], dim=-1)  # (B, merged_dim)
        merged = self.merged_fc(merged)  # (B, 128)
        
        # Policy and value heads:
        pi = self.policy_head(merged)  # (B, 2)
        v = self.value_head(merged) # (B, 1)

        return F.log_softmax(pi, dim=1), v
