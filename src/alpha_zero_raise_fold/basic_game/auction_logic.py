import numpy as np
from enum import Enum
from scipy.stats import gamma
import pdb
NUM_SIMS = 10000
shape = 2.5
scale = 20

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
            self.rand = np.random.randint(100000000)

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
            return hash(self.rand)

        def __eq__(self, other):
            if isinstance(other, AuctionEnv.Athlete):
                return (self.rand) == (other.rand)
            return False


        def __lt__(self, other):
            if self.position != other.position:
                position_order = {AuctionEnv.Position.FORWARD: 0, AuctionEnv.Position.DEFENSEMAN: 1, AuctionEnv.Position.GOALIE: 2}
                return position_order[self.position] < position_order[other.position]
            return self.mean < other.mean
        
    def pop_random(s):
        # Choose a random element by converting the set to a list or tuple
        elem = np.random.choice(tuple(s))
        s.remove(elem)
        return elem
        
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

    def __init__(self, num_players=6, budget=50, num_forwards=4, num_defensemen=2, num_goalies=1, state=None):
        if state:

            self.num_players = state["NUM_PLAYERS"]
            self.GAME_BUDGET = state["GAME_BUDGET"]
            self.FORWARDS_NEEDED = state["FORWARDS_NEEDED"]
            self.DEFENSEMEN_NEEDED = state["DEFENSEMEN_NEEDED"]
            self.GOALIES_NEEDED = state["GOALIES_NEEDED"]

            self.athletes = []
            for pos, means in state["athletes_left"].items():
                for mean in means:
                    position = self.Position[pos.upper()]
                    self.athletes.append(self.Athlete(mean=mean, position=position))

            self.archive_athletes = self.athletes.copy()
            self.forwards_left = state["forwards_left"]
            self.defense_left = state["defense_left"]
            self.goalies_left = state["goalies_left"]
            self.rounds_left = len(self.athletes)

            self.budgets = np.array(state["budgets"])
            self.bidders_left = np.array(state["bidders_left"])
            self.members_forwards_needed = np.array(state["members_forwards_needed"])
            self.member_defense_needed = np.array(state["member_defense_needed"])
            self.member_goalies_needed = np.array(state["member_goalies_needed"])

            self.nominated_player = self.Athlete(mean=state["nominated_player"]["mean"], position=self.Position[state["nominated_player"]["position"].upper()])
            self.current_bid = state["current_bid"]
            self.current_bidder = np.argmax(state["current_bidder"])

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

            np.random.shuffle(self.athletes)

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

            self.bidders_left = np.ones(self.num_players)

            self.members_forwards_needed = np.ones(self.num_players) * self.FORWARDS_NEEDED
            self.member_defense_needed = np.ones(self.num_players) * self.DEFENSEMEN_NEEDED
            self.member_goalies_needed = np.ones(self.num_players) * self.GOALIES_NEEDED

            self.nominated_player = AuctionEnv.pop_random(self.athletes)
            self.current_bid = 0
            self.current_bidder = np.random.randint(self.num_players)

            self.members_means = np.zeros(self.num_players)

            self.round_over = False
            self.game_over = False

            self.history = {} # Here we will store who each player went to and for how much

    def get_valid_moves(self):
        ''' 
        Calculating what moves are valid - shd speed up MCTS I think
        
        '''

        valid = np.array([1,1]) #by default can do both

        if (self.nominated_player.position == self.Position.FORWARD) and (self.forwards_left == self.members_forwards_needed[self.current_bidder]):
            valid[0] = 0
        elif (self.nominated_player.position == self.Position.DEFENSEMAN) and (self.defense_left == self.member_defense_needed[self.current_bidder]):
            valid[0] = 0
        elif (self.nominated_player.position == self.Position.GOALIE) and (self.goalies_left == self.member_goalies_needed[self.current_bidder]):
            valid[0] = 0

        #Check one - make sure we have the budget to bid up
        if self.budgets[self.current_bidder] - self.members_forwards_needed[self.current_bidder] - self.member_defense_needed[self.current_bidder] - self.member_goalies_needed[self.current_bidder] < self.current_bid:
            valid[1] = 0
        #check two - make sure we have roster space for this athlete
        if self.nominated_player.position == self.Position.FORWARD:
            if self.members_forwards_needed[self.current_bidder] == 0:
                valid[1] = 0
        elif self.nominated_player.position == self.Position.DEFENSEMAN:
            if self.member_defense_needed[self.current_bidder] == 0:
                valid[1] = 0
        elif self.nominated_player.position == self.Position.GOALIE:
            if self.member_goalies_needed[self.current_bidder] == 0:
                valid[1] = 0

        if np.sum(valid) == 0:
            raise RuntimeError(f"IN VALID ACTIONS: current_bidder is {self.current_bidder} {self.get_state()}")
        return valid

    def play_action(self, move):

        if self.bidders_left[self.current_bidder] == 0:
            raise RuntimeError("really shouldn't be here")
            while self.bidders_left[self.current_bidder] == 0:
                self.current_bidder = (self.current_bidder + 1) % self.num_players
            return



        if move not in [0,1]:
            raise ValueError(f"Invalid action. {move} is not a valid action - completely off")
        elif self.get_valid_moves()[move] == 0:
            raise ValueError(f"Illegal action. {move} is not a valid action. Valid moves are {self.get_valid_moves()}")
        if move == 0:
            # Pass
            if np.sum(self.bidders_left) == 1:
                # someone else has won the player
                if np.sum(self.bidders_left) == 1:
                    winner = np.where(self.bidders_left == 1)[0][0]
                else:
                    winner = self.current_bidder
                self.budgets[winner] -= self.current_bid
                self.history[self.nominated_player] = { "price" : self.current_bid, "winner" : winner }
                #print(f"{len(self.history)} Players Won:             {self.nominated_player} has been won by {winner} for {self.current_bid}")
                self.current_bid = 0
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
                else:
                    self.nominated_player = AuctionEnv.pop_random(self.athletes)
                    self.bidders_left = np.ones(self.num_players)
                    for i in range(self.num_players):
                        if self.nominated_player.position == self.Position.FORWARD:
                            if self.members_forwards_needed[i] == 0:
                                self.bidders_left[i] = 0
                        elif self.nominated_player.position == self.Position.DEFENSEMAN:
                            if self.member_defense_needed[i] == 0:
                                self.bidders_left[i] = 0
                        elif self.nominated_player.position == self.Position.GOALIE:
                            if self.member_goalies_needed[i] == 0:
                                self.bidders_left[i] = 0

                    assert np.sum(self.bidders_left) > 0, "No bidders left for this player"
            else:
                self.bidders_left[self.current_bidder] = 0
            self.current_bidder = (self.current_bidder + 1) % self.num_players
            cnt = 0
            while self.bidders_left[self.current_bidder] == 0:
                cnt += 1
                if cnt > self.num_players + 3:
                    raise RuntimeError(f"help smth wrong {self.get_state()}")
                self.current_bidder = (self.current_bidder + 1) % self.num_players
        if move == 1:
            # Bid up - or at least try to

            if np.sum(self.bidders_left) <= 1 and self.current_bid > 0:
                self.current_bid -= 1

            self.current_bid += 1
                
            #now update bid and stuff

            if np.sum(self.bidders_left) <= 1:
                # someone else has won the player
                if np.sum(self.bidders_left) == 1:
                    winner = np.where(self.bidders_left == 1)[0][0]
                else:
                    winner = self.current_bidder
                self.budgets[winner] -= self.current_bid
                #print(f"{self.nominated_player} has been won by {winner} for {self.current_bid}")
                self.history[self.nominated_player] = { "price" : self.current_bid, "winner" : winner }
                self.current_bid = 0
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
                    self.nominated_player = AuctionEnv.pop_random(self.athletes)
                    self.bidders_left = np.ones(self.num_players)
                    for i in range(self.num_players):
                        if self.nominated_player.position == self.Position.FORWARD:
                            if self.members_forwards_needed[i] == 0:
                                self.bidders_left[i] = 0
                        elif self.nominated_player.position == self.Position.DEFENSEMAN:
                            if self.member_defense_needed[i] == 0:
                                self.bidders_left[i] = 0
                        elif self.nominated_player.position == self.Position.GOALIE:
                            if self.member_goalies_needed[i] == 0:
                                self.bidders_left[i] = 0
                    assert np.sum(self.bidders_left) > 0, "No bidders left for this player"
            cnt = 1
            self.current_bidder = (self.current_bidder + 1) % self.num_players
            while self.bidders_left[self.current_bidder] == 0:
                cnt += 1
                self.current_bidder = (self.current_bidder + 1) % self.num_players

                if cnt > self.num_players + 3:
                    pdb.set_trace()
                
            
        

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
            "FORWARDS_SCLAE" : scale,
            "GOALIES_SHAPE" : shape,
            "GOALIES_SCLAE" : scale,
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
            "current_bid": self.current_bid,
            "current_bidder": np.eye(self.num_players)[self.current_bidder],
            "bidders_left": self.bidders_left,
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
            },
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