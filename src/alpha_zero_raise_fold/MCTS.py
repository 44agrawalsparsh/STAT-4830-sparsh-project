import logging
import math

import numpy as np
import random

EPS = 1e-8

log = logging.getLogger(__name__)


class VanillaMCTS():

    def __init__(self, game, num_simulations_per_action=10):
        self.game = game
        self.num_sims = num_simulations_per_action

    def random_playout(self, state):
        """
        Simulate a random game (playout) from the given state until termination.
        
        Args:
            game: The game environment object.
            state: The current state from which to start the simulation.
        
        Returns:
            A reward (or reward vector) representing the final outcome of the game.
        """
        current_state = state
        while True:
            # Check if the game has ended.
            result = self. game.getGameEnded(current_state)
            # Here we assume that if the game is not ended, getGameEnded returns 0.
            # When the game ends, it returns a nonzero value (or a reward vector).
            if type(result) is not int:
                return result

            # Get valid moves and choose one at random.
            valid_moves = self.game.getValidMoves(current_state)
            valid_actions = [i for i, valid in enumerate(valid_moves) if valid == 1]
            # Randomly select an action among valid ones.
            action = random.choice(valid_actions)
            current_state = self.game.getNextState(current_state, action)

    def vanilla_monte_carlo(self, state):
        """
        Perform a vanilla Monte Carlo search from the given state.
        
        For each valid action from the current state, perform a number of random
        simulations until the game terminates. Average the reward(s) over all simulations
        for each action and return the result.
        
        Args:
            game: The game environment object.
            state: The current state from which to perform the Monte Carlo search.
            num_simulations_per_action (int): Number of simulations to run for each valid action.
        
        Returns:
            A policy (deterministic choosing the better action), value (reward vector for said policy)
        """
        # Get the valid moves for the current state.
        valid_moves = self.game.getValidMoves(state)
        # Prepare a dictionary to collect simulation outcomes for each valid action.
        action_results = {action: [] for action, valid in enumerate(valid_moves) if valid == 1}

        current_player = self.game.getCurrentPlayer(state)

        # For each valid action, run the specified number of simulations.
        for action in action_results:
            for _ in range(self.num_sims):
                # First, apply the action to get the next state.
                next_state = self.game.getNextState(state, action)
                # Run a random playout from the resulting state.
                result = self.random_playout(next_state)
                action_results[action].append(result)

        # Compute the average reward for each action.
        average_rewards = {}
        for action, rewards in action_results.items():
            # Convert list of rewards to a NumPy array to compute the mean.
            rewards_array = np.array(rewards)
            # Here we assume rewards are vectors (e.g., one entry per player).
            # If rewards are scalars, np.mean still works.
            avg_reward = np.mean(rewards_array, axis=0)
            average_rewards[action] = avg_reward

        all_actions = list(average_rewards.keys())
        all_actions.sort(key = lambda x : average_rewards[x][current_player], reverse=True)

        pi = np.eye(self.game.getActionSize())[all_actions[0]]
        v = average_rewards[all_actions[0]]
        return pi, v

    # Example usage:
    # Assuming `game` is your game environment and `current_state` is a valid state:
    #
    # avg_rewards = vanilla_monte_carlo(game, current_state, num_simulations_per_action=100)
    # print("Average rewards for each valid action:", avg_rewards)



class MCTS():
    """
    This class handles the MCTS tree.
    """

    def __init__(self, game, nnet, args):
        self.game = game
        self.nnet = nnet
        self.args = args
        self.Qsa = {}  # stores Q values for s,a (as defined in the paper)
        self.Nsa = {}  # stores #times edge s,a was visited
        self.Ns = {}  # stores #times board s was visited
        self.Ps = {}  # stores initial policy (returned by neural net)

        self.Es = {}  # stores game.getGameEnded ended for board s
        self.Vs = {}  # stores game.getValidMoves for board s

        #self.terminal_seen = 0
        #self.terminal_goal = -1

        self.search_called = 0

    def getActionProb(self, state, temp=1):
        """
        This function performs numMCTSSims simulations of MCTS starting from
        state.

        Returns:
            probs: a policy vector where the probability of the ith action is
                   proportional to Nsa[(s,a)]**(1./temp)
        """

        #self.terminal_goal = self.terminal_seen + self.args.minTerm
        i = 0

        #original_player = self.game.getCurrentPlayer(state)
        while i < self.args.numMCTSSims:
            #for i in range(self.args.numMCTSSims):
            i += 1
            if i % 100 == 0:
                print(f"Sim: {i}")
            self.search(state)

        s = self.game.stringRepresentation(state)
        counts = [self.Nsa[(s, a)] if (s, a) in self.Nsa else 0 for a in range(self.game.getActionSize())]

        if temp == 0:
            bestAs = np.array(np.argwhere(counts == np.max(counts))).flatten()
            bestA = np.random.choice(bestAs)
            probs = [0] * len(counts)
            probs[bestA] = 1
            return probs

        counts = [x ** (1. / temp) for x in counts]
        counts_sum = float(sum(counts))
        probs = [x / counts_sum for x in counts]
        return probs
    
    def search(self, state):
        ''' 
        Modifying the below search to work for our n-dimensional game

        Basically we don't care about canonical form. That's just something that is passed in to the nn
        
        '''

        #self.search_called += 1 
        #print(self.search_called)


        s = self.game.stringRepresentation(state)
        cur_player = self.game.getCurrentPlayer(state)

        if s not in self.Es:
            self.Es[s] = self.game.getGameEnded(state) #should be 0 if game hasn't ended
        if type(self.Es[s]) is not int:
            # terminal node
            #self.terminal_seen += 1
            #print(self.terminal_seen)
            return self.Es[s]
        
        nn_pi = 0

        if s not in self.Ps:
            # leaf node -> idk if this is true but okie
            nn_pi, v = self.nnet.predict(state)
            #print(nn_pi)
            #print(nn_pi, "PREDS")
            valids = self.game.getValidMoves(state)
            self.Ps[s] = nn_pi * valids  # masking invalid moves
            sum_Ps_s = np.sum(self.Ps[s])
            if sum_Ps_s > 0:
                self.Ps[s] /= sum_Ps_s  # renormalize
                #print(self.Ps[s], "renormalize")
            else:
                # if all valid moves were masked make all valid moves equally probable

                # NB! All valid moves may be masked if either your NNet architecture is insufficient or you've get overfitting or something else.
                # If you have got dozens or hundreds of these messages you should pay attention to your NNet and/or training process.   
                log.error(f"All valid moves were masked, doing a workaround. NN output = {self.Ps[s]}")
                self.Ps[s] = self.Ps[s] + valids
                self.Ps[s] /= np.sum(self.Ps[s])

            self.Vs[s] = valids
            self.Ns[s] = 0
            #if self.terminal_seen > self.terminal_goal:
            return v #have to figure out how this works

        valids = self.Vs[s]
        cur_best = -float('inf')
        best_act = -1

        # pick the action with the highest upper confidence bound
        for a in range(self.game.getActionSize()):
            if valids[a]:
                if (s, a) in self.Qsa:
                    u = self.Qsa[(s, a)] + self.args.cpuct * self.Ps[s][a] * math.sqrt(self.Ns[s]) / (
                            1 + self.Nsa[(s, a)])
                else:
                    u = self.args.cpuct * self.Ps[s][a] * math.sqrt(self.Ns[s] + EPS)  # Q = 0 ?

                if u > cur_best:
                    cur_best = u
                    best_act = a
        #print(valids, a, u, )

        a = best_act

        if valids[a] != self.game.getValidMoves(state)[a]:
            print(valids, self.game.getValidMoves(state))
            print(s == self.game.stringRepresentation(state))
            raise RuntimeError("Mismatched valids!!!!")

        if valids[a] == 0:
            print(valids, a, u, nn_pi, self.Ps[a])
            raise RuntimeError("invalid selected somehow")
        else:
            next_state = self.game.getNextState(state, a)
        #do we need next player # nextPlayer = self.game.getCurrentPlayer(next_state)

        v = self.search(next_state)
        if (s, a) in self.Qsa:
            self.Qsa[(s, a)] = (self.Nsa[(s, a)] * self.Qsa[(s, a)] + v[cur_player]) / (self.Nsa[(s, a)] + 1)
            self.Nsa[(s, a)] += 1

        else:
            self.Qsa[(s, a)] = v[cur_player]
            self.Nsa[(s, a)] = 1

        self.Ns[s] += 1
        return v