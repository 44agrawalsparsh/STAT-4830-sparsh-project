from __future__ import print_function
import sys
sys.path.append('..')
sys.path.append('../..')
from Game import Game
from .auction_logic import AuctionEnv
import numpy as np
import itertools
import copy
from .neural_networks.formatter import prepare_batch_state
class BasicAuctionGame(Game):


    def __init__(self, num_players=6, budget=50, num_forwards=4, num_defensemen=2, num_goalies=1):
        self.num_players = num_players
        self.budget = budget
        self.num_forwards = num_forwards
        self.num_defensemen = num_defensemen
        self.num_goalies = num_goalies

    def getInitState(self):
        # return initial board (numpy board)
        game = AuctionEnv(self.num_players, self.budget, self.num_forwards, self.num_defensemen, self.num_goalies)
        return game.get_state()
    
    def getStructure(self):
        structure = {
            "NUM_PLAYERS" : self.num_players,
            "GAME_BUDGET" : self.budget,
            "FORWARDS_NEEDED" : self.num_forwards,
            "DEFENSEMEN_NEEDED" : self.num_defensemen,
            "GOALIES_NEEDED" : self.num_goalies
        }
        return structure


    def getActionSize(self):
        # return number of actions
        return 2 #raise, fold

    def getNextState(self, state, action):
        # wrapper for play_action
        game = AuctionEnv(state=state)
        game.play_action(action)
        return game.get_state()

    def getValidMoves(self, state):
        # return a fixed size binary vector
        game = AuctionEnv(state=state)
        return game.get_valid_moves()

    def getGameEnded(self, state):
        # return game reward function if game over else 0
        
        game = AuctionEnv(state=state)
        try:
            return game.get_game_score()
        except:
            return 0
        

    def getCanonicalForm(self, state, player):
        new_state = copy.deepcopy(state)
        if player == 0:
            return new_state
        def switch(arr):
            player_val = arr[player]
            other_val = arr[0]
            arr[player] = other_val
            arr[0] = player_val
        switch(new_state["members_means"])
        switch(new_state["budgets"])
        switch(new_state["members_forwards_needed"])
        switch(new_state["member_goalies_needed"])
        switch(new_state["current_bidder"])
        switch(new_state["bidders_left"])
        for win in list(new_state["history"].values()):
            if win["winner"] == 0:
                win["winner"] = player 
            elif win["winner"] == player:
                win["winner"] = 0
        return new_state

    
    def getCurrentPlayer(self, state):
        return np.argmax(state["current_bidder"])
            

    def getSymmetries(self, state, pi, cur_player):

        raise RuntimeError("This function is deprecated - we process all symmetries as one")
        # scramble all players except for cur_player

        def swap_players(p1, p2, state):
            def switch(arr):
                p1_val = arr[p1]
                p2_val = arr[p2]

                arr[p1] = p2_val
                arr[p2] = p1_val
            
            switch(state["members_means"])
            switch(state["budgets"])
            switch(state["members_forwards_needed"])
            switch(state["member_goalies_needed"])
            switch(state["current_bidder"])
            switch(state["bidders_left"])

            for win in list(state["history"].values()):
                if win["winner"] == p2:
                    win["winner"] = p1 
                elif win["winner"] == p1:
                    win["winner"] = p2
            
            return state
        
        #Go through all permuations
        def fixed_one_permutations(n):
            # Generate all permutations of elements excluding cur_player
            for perm in itertools.permutations([i for i in range(n) if i != cur_player]):
                # Insert cur_player at the second last position of each permutation
                yield perm[:n-2] + (cur_player,) + perm[n-2:]

        symmetries = []
        for perm in fixed_one_permutations(self.num_players):
            print("Processing permutation:", perm)
            new_state = copy.deepcopy(state)
            
            mapping = [p for p in perm]

            # Now compute the cycles of the permutation mapping.
            visited = [False] * self.num_players
            for i in range(self.num_players):
                # If already visited or already in place, skip.
                if visited[i] or mapping[i] == i:
                    continue
                cycle = []
                j = i
                while not visited[j]:
                    visited[j] = True
                    cycle.append(j)
                    j = mapping[j]
                
                # For each cycle of length L (L > 1), we need L - 1 swaps.
                # A simple method is to fix the first index and swap it with all others in the cycle.
                for k in range(1, len(cycle)):
                    swap_players(cycle[0], cycle[k], new_state)

            symmetries.append((new_state, pi))
        
        return symmetries

    def stringRepresentation(self, state):
        return str(state)
    
    def prepare_tensor_input(self, states, device):
        ''' 
        states is list of game states we want as an input
        '''

        new_states = []
        for state in states:
            cur_bidder = self.getCurrentPlayer(state)
            canonical = self.getCanonicalForm(state, cur_bidder)
            new_states.append(canonical)

        return prepare_batch_state(new_states, device=device)
