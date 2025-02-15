import numpy as np
from enum import Enum
from scipy.stats import gamma
import pdb
NUM_SIMS = 10000
shape = 2.5
scale = 20
import random


class AuctionEnv:
    """ 

     Project Description:

    My project idea focuses on solving a dynamic decision-making problem in the context of a simplified fantasy hockey auction. The setup for the auction is as follows:

    League Structure: The league consists of 8 members, each provided with 1000 tokens to spend in an auction.

    Team Composition: Each member must build a team of 20 players: 12 forwards, 6 defensemen, and 2 goalies.

    Player Valuation: For simplicity's sake each player’s value is represented as independent random variables with a known mean and variance - all members agree upon these distributions. These values are realized after the auction is completed.

    Auction Mechanics:

    Players are randomly nominated for auction each round.

    The bidding proceeds cyclically, where each member decides whether to bid up or pass.

    Bidding up increases the current price by one token and temporarily assigns the player to the bidder.

    If a member passes, they are no longer allowed to bid on the player for that round.

    The auction ends for a player when only one member remains in the bidding, and the player is awarded to them for the final price.

    Objective: My overarching goal is to maximize E(rank), determined by the cumulative realized value of the drafted players compared to other members. This creates a complex strategic environment where decisions need to account for budget constraints, remaining roster spots, and the uncertainty in player valuations.
        
    Below we define the environment class for the auction game. The environment is responsible for managing the game state, executing actions, and providing feedback to the agents.

    ** CHANGES **

    Now making it a vickrey auction. So each player submits a bid and the highest bid pays the price of player 2. I will also pre randomize the order in which players will be nominated. 

    This won't be visible (as won't the players other bids) but it'll ensure the game tree is simpler without much loss of generality
    
    
    """


class AuctionEnv:
    """
    
    Project Description:

    My project idea focuses on solving a dynamic decision-making problem in the context of a simplified fantasy hockey auction. The setup for the auction is as follows:

    League Structure: The league consists of 8 members, each provided with 1000 tokens to spend in an auction.

    Team Composition: Each member must build a team of 20 players: 12 forwards, 6 defensemen, and 2 goalies.

    Player Valuation: For simplicity's sake each player’s value is represented as independent random variables with a known mean and variance - all members agree upon these distributions. These values are realized after the auction is completed.

    Auction Mechanics:

    Players are randomly nominated for auction each round.

    The bidding proceeds cyclically, where each member decides whether to bid up or pass.

    Bidding up increases the current price by one token and temporarily assigns the player to the bidder.

    If a member passes, they are no longer allowed to bid on the player for that round.

    The auction ends for a player when only one member remains in the bidding, and the player is awarded to them for the final price.

    Objective: My overarching goal is to maximize E(rank), determined by the cumulative realized value of the drafted players compared to other members. This creates a complex strategic environment where decisions need to account for budget constraints, remaining roster spots, and the uncertainty in player valuations.
        
    Below we define the environment class for the auction game. The environment is responsible for managing the game state, executing actions, and providing feedback to the agents.

    ** CHANGES **

    Now making it a vickrey auction. So each player submits a bid and the highest bid pays the price of player 2. I will also pre randomize the order in which players will be nominated. 

    This won't be visible (as won't the players other bids) but it'll ensure the game tree is simpler without much loss of generality
    
    Will also add two new methods:

    So we have get_state which lets us create more games from the dict - sure

    We also will have get_observation - this takes in a player and shows whats visible to them - will use this to generate the tesnor for nn for example

    We also will have a get_hash - this rounds some of the values in the game and returns an string then characterizing an observation. Used for MCTS to simplify the game further

    """

    class Position(Enum):
        GOALIE = "goalie"
        FORWARD = "forward"
        DEFENSEMAN = "defenseman"
    
    class Athlete:
        def __init__(self, mean, position, owner=-1):
            self.mean = mean
            self.position = position
            self.owner = owner

            pos_code = "G"
            if self.position == AuctionEnv.Position.FORWARD:
                pos_code == "F"
            elif self.position == AuctionEnv.Position.DEFENSEMAN:
                pos_code == "D"
            
            self.hash = f"{pos_code}-{self.mean}"
        
            

        def has_owner(self):
            return self.owner >= 0
        
        def set_owner(self, owner):
            self.owner = owner

        def __repr__(self):
            if self.owner >= 0:
                return f"Athlete(mean={self.mean}, position={self.position}, owner={self.owner})"
            else:
                return f"Athlete(mean={self.mean}, position={self.position})"

        def __hash__(self):
            return hash(self.hash)

        def __eq__(self, other):
            if isinstance(other, AuctionEnv.Athlete):
                return (self.hash) == (other.hash)
            return False


        def __lt__(self, other):
            if self.position != other.position:
                position_order = {AuctionEnv.Position.FORWARD: 0, AuctionEnv.Position.DEFENSEMAN: 1, AuctionEnv.Position.GOALIE: 2}
                return position_order[self.position] < position_order[other.position]
            return self.mean < other.mean
        
    '''    
    def pop_random(s):
        # Choose a random element by converting the set to a list or tuple
        elem = np.random.choice(tuple(s))
        s.remove(elem)
        return elem
    '''

    def generate_athletes(num_players, num_forwards, num_defensemen, num_goalies, shape, scale):
        """
        Generates a list of athletes with random mean and variance for each position type
        """
        athletes = []
        for _ in range(num_forwards * num_players):
            mean = np.round(gamma.rvs(a=shape, scale=scale),2)
            athletes.append(AuctionEnv.Athlete(mean=mean, position=AuctionEnv.Position.FORWARD))
        for _ in range(num_defensemen * num_players):
            mean = np.round(gamma.rvs(a=shape, scale=scale*0.55),2)
            athletes.append(AuctionEnv.Athlete(mean=mean, position=AuctionEnv.Position.DEFENSEMAN))
        for _ in range(num_goalies * num_players):
            mean = np.round(gamma.rvs(a=shape, scale=scale), 2)
            athletes.append(AuctionEnv.Athlete(mean=mean, position=AuctionEnv.Position.GOALIE))
        return athletes

    def __init__(self, num_players=2, budget=1000, num_forwards=5, num_defensemen=3, num_goalies=1, state=None):
        if state:
            self.num_players = state["NUM_PLAYERS"]
            self.GAME_BUDGET = state["GAME_BUDGET"]
            self.FORWARDS_NEEDED = state["FORWARDS_NEEDED"]
            self.DEFENSEMEN_NEEDED = state["DEFENSEMEN_NEEDED"]
            self.GOALIES_NEEDED = state["GOALIES_NEEDED"]

            self.athletes = []
            for athlete in state["athletes_left"]:
                position = self.Position[athlete["position"].upper()]
                mean = athlete["mean"]
                self.athletes.append(self.Athlete(mean=mean, position=position))


            self.archive_athletes = []
            for pos, means in state["athletes_starting"].items():
                for mean in means:
                    position = self.Position[pos.upper()]
                    self.archive_athletes.append(self.Athlete(mean=mean, position=position))

            self.forwards_left = state["forwards_left"]
            self.defense_left = state["defense_left"]
            self.goalies_left = state["goalies_left"]
            self.rounds_left = len(self.athletes)

            self.budgets = np.array(state["budgets"])
            self.bids_placed = np.array(state["current_bids"])
            self.current_bidder = np.argmax(state["current_bidder"])

            self.members_forwards_needed = np.array(state["members_forwards_needed"])
            self.member_defense_needed = np.array(state["member_defense_needed"])
            self.member_goalies_needed = np.array(state["member_goalies_needed"])

            self.nominated_player = self.Athlete(mean=state["nominated_player"]["mean"], position=self.Position[state["nominated_player"]["position"].upper()])

            self.members_means = np.array(state["members_means"])

            self.round_over = False
            self.game_over = False

            self.history = {}
            for i, player_info in state["history"].items():
                position = self.Position[player_info["position"].upper()]
                player = self.Athlete(mean=player_info["mean"], position=position)
                player.set_owner(player_info["winner"])
                self.history[player] = {"price": player_info["price"], "winner": player_info["winner"]}

            if self.forwards_left + self.defense_left + self.goalies_left == 0:
                self.game_over = True
        else:
            self.num_players = num_players
            self.GAME_BUDGET = budget
            self.FORWARDS_NEEDED = num_forwards
            self.DEFENSEMEN_NEEDED = num_defensemen
            self.GOALIES_NEEDED = num_goalies

            self.athletes = AuctionEnv.generate_athletes(self.num_players, self.FORWARDS_NEEDED, self.DEFENSEMEN_NEEDED, self.GOALIES_NEEDED, shape, scale)
            self.archive_athletes = self.athletes.copy()
            
            random.shuffle(self.athletes)

            # Sanity check players are valid
            empiric_forwards = np.sum([athlete.position == self.Position.FORWARD for athlete in self.athletes])
            empiric_goalies = np.sum([athlete.position == self.Position.GOALIE for athlete in self.athletes])
            empiric_defense = np.sum([athlete.position == self.Position.DEFENSEMAN for athlete in self.athletes])

            assert np.sum([athlete.has_owner() for athlete in self.athletes]) == 0, "Players cannot have owners at initialization"
            assert empiric_forwards == num_forwards*self.num_players, f"Number of forwards does not match player list. Expected {num_forwards*self.num_players} see {empiric_forwards}"
            assert empiric_goalies == num_goalies*self.num_players, f"Number of goalies does not match player list. Expected {num_goalies*self.num_players} see {empiric_goalies}"
            assert empiric_defense == num_defensemen*self.num_players, f"Number of defensemen does not match player list. Expected {num_defensemen*self.num_players} see {empiric_defense}"

            self.forwards_left = self.num_players * self.FORWARDS_NEEDED
            self.defense_left = self.num_players * self.DEFENSEMEN_NEEDED
            self.goalies_left = self.num_players * self.GOALIES_NEEDED
            self.rounds_left = len(self.athletes)

            self.budgets = np.ones(self.num_players) * self.GAME_BUDGET

            #self.bidders_left = np.ones(self.num_players)
            self.bids_placed = -1 * np.ones(self.num_players) #WON"T BE VISIBLE
            self.current_bidder = 0 #vickrey so dont need randomization

            self.members_forwards_needed = np.ones(self.num_players) * self.FORWARDS_NEEDED
            self.member_defense_needed = np.ones(self.num_players) * self.DEFENSEMEN_NEEDED
            self.member_goalies_needed = np.ones(self.num_players) * self.GOALIES_NEEDED

            self.nominated_player = self.athletes.pop()
            
            self.members_means = np.zeros(self.num_players)

            self.round_over = False
            self.game_over = False

            self.history = {} # Here we will store who each player went to and for how much

    
    def get_must_bid(self, player=None):
        ''' 
        
        Determines whether we are forced bidders or not
        
        '''

        if player == None:
            player = self.current_bidder

        if (self.nominated_player.position == self.Position.FORWARD) and (self.forwards_left == self.members_forwards_needed[player]):
            return 1
        elif (self.nominated_player.position == self.Position.DEFENSEMAN) and (self.defense_left == self.member_defense_needed[player]):
            return 1
        elif (self.nominated_player.position == self.Position.GOALIE) and (self.goalies_left == self.member_goalies_needed[player]):
            return 1

        return 0

    def get_max_bid(self, player=None):
        ''' 
        
        Determines the maximum bid for a given player (default to current bidder)
        
        '''

        if player == None:
            player = self.current_bidder

        if self.nominated_player.position == self.Position.FORWARD:
            if self.members_forwards_needed[player] == 0:
                return 0
        elif self.nominated_player.position == self.Position.DEFENSEMAN:
            if self.member_defense_needed[player] == 0:
                return 0
        elif self.nominated_player.position == self.Position.GOALIE:
            if self.member_goalies_needed[player] == 0:
                return 0
            
        max_val = self.budgets[player] - self.members_forwards_needed[player] - self.member_defense_needed[player] - self.member_goalies_needed[player] + 1
        return max_val
    
    def sanitize_moves(self, pi):

        if self.get_must_bid():
            pi[1] += 1e-2

        if self.get_max_bid() == 0:
            pi = np.zeros(len(pi))
            pi[0] = 1 #just so we learn in these cases too
        
        pi /= np.sum(pi)

        return pi


    
    def play_action(self, action):
        ''' 
        
        Must enter self.current_player here
        
        '''
        #pdb.set_trace()

        min_bid = self.get_must_bid()
        max_bid = self.get_max_bid()

        assert max_bid >= 0, f"Max bid is {max_bid}"
        assert action <= 10, f"action is {action}"
        assert max_bid <= self.budgets[self.current_bidder]
        assert min_bid <= max_bid, f"Max bid is {max_bid}, min is {min_bid}, state is {self.get_state()}"

        bid = action/10*(max_bid - min_bid) + min_bid
        bid = np.round(bid)

        self.bids_placed[self.current_bidder] = bid

        next_player = (self.current_bidder + 1) % self.num_players

        if next_player == 0:
            ### means we have determined the winner
            try:
                assert np.sum(self.bids_placed) >= 0
            except Exception as e:
                print(self.get_state())
                assert np.sum(self.bids_placed) >= 0

            highest_bid_indices = np.where(self.bids_placed == np.max(self.bids_placed))[0]
            winner = np.random.choice(highest_bid_indices)
            price_paid = np.partition(self.bids_placed, -2)[-2]

            self.budgets[winner] -= price_paid
            #print(f"{self.nominated_player} has been won by {winner} for {self.current_bid}")
            self.history[self.nominated_player] = { "price" : price_paid, "winner" : winner }
            self.bids_placed = -1 * np.ones(self.num_players)
            self.nominated_player.set_owner(winner)

            self.members_means[winner] += self.nominated_player.mean

            if self.nominated_player.position == self.Position.FORWARD:
                self.members_forwards_needed[winner] -= 1
                self.forwards_left -= 1
            elif self.nominated_player.position == self.Position.DEFENSEMAN:
                self.member_defense_needed[winner] -= 1
                self.defense_left -= 1
            elif self.nominated_player.position == self.Position.GOALIE:
                self.member_goalies_needed[winner] -= 1
                self.goalies_left -= 1

            if len(self.athletes) == 0:
                self.game_over = True
                return
            else:
                self.nominated_player = self.athletes.pop()
        
        self.current_bidder = next_player

    def get_observation(self):
        ### Return the observable state for self.current_bidder
        ''' 

        What this entails:
            Current mean and variance of each player's roster
            Current budget of each player
            Current requirements of each player
            Remaining forwards, defensemen, and goalies
            Current player being auctioned
            Current bid
            Current bidder - one hot encoded
            Which bidders are left - one hot encoded as rn
            Needs of each player - num forwards needed, num defensemen needed, num goalies needed
        '''
        def pos_to_string(pos):
            if pos == self.Position.FORWARD:
                return "forward"
            elif pos == self.Position.DEFENSEMAN:
                return "defenseman"
            elif pos == self.Position.GOALIE:
                return "goalie"
        

        obs = {
            "FORWARDS_SHAPE" : shape,
            "FORWARDS_SCALE" : scale,
            "GOALIES_SHAPE" : shape,
            "GOALIES_SCALE" : scale,
            "DEFENSE_SHAPE" : shape,
            "DEFENSE_SCALE" : 0.55*scale,
            "NUM_PLAYERS" : self.num_players,
            "GAME_BUDGET" : self.GAME_BUDGET,
            "FORWARDS_NEEDED" : self.FORWARDS_NEEDED,
            "DEFENSEMEN_NEEDED" : self.DEFENSEMEN_NEEDED,
            "GOALIES_NEEDED" : self.GOALIES_NEEDED,
            "PLAYER_ID" : self.current_bidder,
            "members_means": self.members_means,
            "budgets": self.budgets,
            "members_forwards_needed": self.members_forwards_needed,
            "member_defense_needed": self.member_defense_needed,
            "member_goalies_needed": self.member_goalies_needed,
            "forwards_left": self.forwards_left,
            "defense_left": self.defense_left,
            "goalies_left": self.goalies_left,
            "nominated_player": {
                "mean" : self.nominated_player.mean,
                "position" : pos_to_string(self.nominated_player.position)
            },
            "athletes_left" : {
                "forward": sorted(
                [athlete.mean for athlete in self.athletes if athlete.position == self.Position.FORWARD],
                key=lambda x: x,
                reverse=True
                ),
                "defenseman": sorted(
                [athlete.mean for athlete in self.athletes if athlete.position == self.Position.DEFENSEMAN],
                key=lambda x: x,
                reverse=True
                ),
                "goalie": sorted(
                [athlete.mean for athlete in self.athletes if athlete.position == self.Position.GOALIE],
                key=lambda x: x,
                reverse=True
                )
            }
        }

        return obs
    
    def get_hash(self):
        ''' 
    
        Simplified version of the game to hash - should make the MCTS explore more without giving up too much
        
        '''

        output = self.get_observation()

        return str(output)

        def budget_round(x):
            if x < 25:
                return x
            elif x < 100:
                return 5*np.round(x/5)
            elif x < 250:
                return 10*np.round(x/10)
            return 25*np.round(x/25)
            

        #Round member means - to nearest 5
        new_means = [5*np.round(x/5) for x in output["members_means"]]
        output["members_means"] = new_means
        new_budgets = [budget_round(x) for x in output["budgets"]]
        output["budgets"] = new_budgets
        output["nominated_player"]["mean"] = 5*np.round(output["nominated_player"]["mean"]/5)
        for pos in output["athletes_left"]:
            output["athletes_left"][pos] = [5*np.round(x/5) for x in output["athletes_left"][pos]]
        
        return str(output)



        

    def get_state(self):
        ### Return the state of the game
        ''' 

        What this entails:
            Current mean and variance of each player's roster
            Current budget of each player
            Current requirements of each player
            Remaining forwards, defensemen, and goalies
            Current player being auctioned
            Current bid
            Current bidder - one hot encoded
            Which bidders are left - one hot encoded as rn
            Needs of each player - num forwards needed, num defensemen needed, num goalies needed
        '''
        def pos_to_string(pos):
            if pos == self.Position.FORWARD:
                return "forward"
            elif pos == self.Position.DEFENSEMAN:
                return "defenseman"
            elif pos == self.Position.GOALIE:
                return "goalie"
        
        

        state = {
            "FORWARDS_SHAPE" : shape,
            "FORWARDS_SCALE" : scale,
            "GOALIES_SHAPE" : shape,
            "GOALIES_SCALE" : scale,
            "DEFENSE_SHAPE" : shape,
            "DEFENSE_SCALE" : 0.55*scale,
            "NUM_PLAYERS" : self.num_players,
            "GAME_BUDGET" : self.GAME_BUDGET,
            "FORWARDS_NEEDED" : self.FORWARDS_NEEDED,
            "DEFENSEMEN_NEEDED" : self.DEFENSEMEN_NEEDED,
            "GOALIES_NEEDED" : self.GOALIES_NEEDED,
            "members_means": self.members_means,
            "budgets": self.budgets,
            "members_forwards_needed": self.members_forwards_needed,
            "member_defense_needed": self.member_defense_needed,
            "member_goalies_needed": self.member_goalies_needed,
            "forwards_left": self.forwards_left,
            "defense_left": self.defense_left,
            "goalies_left": self.goalies_left,
            "nominated_player": {
                "mean" : self.nominated_player.mean,
                "position" : pos_to_string(self.nominated_player.position)
            },
            "current_bids": self.bids_placed,
            "current_bidder": np.eye(self.num_players)[self.current_bidder],
            "athletes_left" : [{"mean" : x.mean, "position" : pos_to_string(x.position)} for x in self.athletes],
            "athletes_starting" : {
                "forward" : [athlete.mean for athlete in self.archive_athletes if athlete.position == self.Position.FORWARD],
                "defenseman" : [athlete.mean for athlete in self.archive_athletes if athlete.position == self.Position.DEFENSEMAN],
                "goalie" : [athlete.mean for athlete in self.archive_athletes if athlete.position == self.Position.GOALIE]
            },
            "history" : {
                i : {"position" : pos_to_string(player.position), "mean" : player.mean, "winner" : self.history[player]["winner"], "price" : self.history[player]["price"]} for i, player in enumerate(self.history)
            }
        }

        return state
    
    def get_game_score(self):
        if self.game_over:
            sorted_indices = np.argsort(self.members_means)
            ranks = np.empty_like(sorted_indices)
            ranks[sorted_indices] = np.arange(1, self.num_players + 1)
            return ranks/np.sum(ranks) #sum 1
        else:
            raise ValueError("Game is not over")