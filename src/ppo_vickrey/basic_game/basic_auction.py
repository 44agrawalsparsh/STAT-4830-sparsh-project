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


    def __init__(self, num_players=2, budget=1000, num_forwards=5, num_defensemen=3, num_goalies=1):
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
        return 2 # mean, variance of distribution

    def getNextState(self, state, action, print_bid=False):
        # wrapper for play_action
        game = AuctionEnv(state=state)
        assert str(state) == str(game.get_state()), f"Received: {state}\n\n\nGenerated: {game.get_state()}"
        game.play_action(action, print_bid=print_bid)
        return game.get_state()
    
    def getObservation(self, state):
        game = AuctionEnv(state=state)
        return game.get_observation()
    
    def getRewards(self, state):
        # return game reward function if game over else 0
        
        game = AuctionEnv(state=state)
        return game.get_reward()
    
    def getGameOver(self, state):
        game = AuctionEnv(state=state)
        return game.game_over
        
    def getCurrentPlayer(self, state):
        return np.argmax(state["current_bidder"])

    def stringRepresentation(self, state):
        game = AuctionEnv(state=state)
        return game.get_hash()
    
    def prepare_tensor_input(self, states, device):
        ''' 
        states is list of game states we want as an input for our NN
        '''

        obs = []
        for state in states:
            game = AuctionEnv(state=state)
            obs.append(game.get_observation())

        return prepare_batch_state(obs, device=device)
