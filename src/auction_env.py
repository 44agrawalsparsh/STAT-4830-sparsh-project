import numpy as np
from enum import Enum
from scipy.stats import gamma
import pdb
NUM_SIMS = 10000

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
        def __init__(self, mean, variance, position, owner=-1):
            self.mean = mean
            self.variance = variance
            self.position = position
            self.owner = owner
            self.rand = np.random.randint(100000000)

            assert self.variance > 0, "Player variance must be positive"

        def has_owner(self):
            return self.owner >= 0
        
        def set_owner(self, owner):
            self.owner = owner

        def __repr__(self):
            if self.owner >= 0:
                return f"Athlete(mean={self.mean}, variance={self.variance}, position={self.position}, owner={self.owner})"
            else:
                return f"Athlete(mean={self.mean}, variance={self.variance}, position={self.position})"

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
            if self.mean != other.mean:
                return self.mean < other.mean
            return self.variance < other.variance

    def __init__(self, athletes, num_players=8, budget=100, num_forwards=12, num_defensemen=6, num_goalies=2):
        self.num_players = num_players
        self.GAME_BUDGET = budget
        self.FORWARDS_NEEDED = num_forwards
        self.DEFENSEMEN_NEEDED = num_defensemen
        self.GOALIES_NEEDED = num_goalies
        self.athletes = athletes
        self.archive_athletes = athletes.copy()

        np.random.shuffle(self.athletes)

        # Sanity check players are valid
        empiric_forwards = np.sum([athlete.position == self.Position.FORWARD for athlete in self.athletes])
        empiric_goalies = np.sum([athlete.position == self.Position.GOALIE for athlete in self.athletes])
        empiric_defense = np.sum([athlete.position == self.Position.DEFENSEMAN for athlete in self.athletes])

        assert np.sum([athlete.has_owner() for athlete in athletes]) == 0, "Players cannot have owners at initialization"
        assert empiric_forwards == num_forwards*self.num_players, f"Number of forwards does not match player list. Expected {num_forwards*self.num_players} see {empiric_forwards}"
        assert empiric_goalies == num_goalies*self.num_players, f"Number of goalies does not match player list. Expected {num_goalies*self.num_players} see {empiric_goalies}"
        assert empiric_defense == num_defensemen*self.num_players, f"Number of defensemen does not match player list. Expected {num_defensemen*self.num_players} see {empiric_defense}"

        self.forwards_left = self.num_players * self.FORWARDS_NEEDED
        self.defense_left = self.num_players * self.DEFENSEMEN_NEEDED
        self.goalies_left = self.num_players * self.GOALIES_NEEDED
        self.rounds_left = len(athletes)

        self.budgets = np.ones(self.num_players) * self.GAME_BUDGET

        self.bidders_left = np.ones(self.num_players)

        self.members_forwards_needed = np.ones(self.num_players) * self.FORWARDS_NEEDED
        self.member_defense_needed = np.ones(self.num_players) * self.DEFENSEMEN_NEEDED
        self.member_goalies_needed = np.ones(self.num_players) * self.GOALIES_NEEDED

        self.nominated_player = self.athletes.pop()
        self.current_bid = 0
        self.current_bidder = np.random.randint(self.num_players)

        self.members_means = np.zeros(self.num_players)
        self.members_variances = np.zeros(self.num_players)

        self.round_over = False
        self.game_over = False

        self.history = {} # Here we will store who each player went to and for how much

    
    def play_action(self, move):

        if self.bidders_left[self.current_bidder] == 0:
            while self.bidders_left[self.current_bidder] == 0:
                self.current_bidder = (self.current_bidder + 1) % self.num_players
            return



        if move not in [0,1]:
            raise ValueError(f"Invalid action. {move} is not a valid action")
        if move == 0:
            # Pass

            #First check to see if we're forced to bid. This happens when the number of this position type left is equal to the number of this position type we need
            if (self.nominated_player.position == self.Position.FORWARD) and (self.forwards_left == self.members_forwards_needed[self.current_bidder]):
                move = 1
            elif (self.nominated_player.position == self.Position.DEFENSEMAN) and (self.defense_left == self.member_defense_needed[self.current_bidder]):
                move = 1
            elif (self.nominated_player.position == self.Position.GOALIE) and (self.goalies_left == self.member_goalies_needed[self.current_bidder]):
                move = 1
            else:
                self.bidders_left[self.current_bidder] = 0
                if np.sum(self.bidders_left) <= 1:
                    # someone else has won the player
                    if np.sum(self.bidders_left) == 1:
                        winner = np.where(self.bidders_left == 1)[0][0]
                    else:
                        winner = self.current_bidder
                    self.budgets[winner] -= self.current_bid
                    self.history[self.nominated_player] = { "price" : self.current_bid, "winner" : winner }
                    self.current_bid = 0
                    self.nominated_player.set_owner(winner)
                    self.members_means[winner] += self.nominated_player.mean
                    self.members_variances[winner] += self.nominated_player.variance

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
                        self.nominated_player = self.athletes.pop()
                        self.bidders_left = np.ones(self.num_players)
                self.current_bidder = (self.current_bidder + 1) % self.num_players
                while self.bidders_left[self.current_bidder] == 0:
                    self.current_bidder = (self.current_bidder + 1) % self.num_players
        if move == 1:
            # Bid up - or at least try to

            self.current_bid += 1

            #Check one - make sure we have the budget to bid up
            if self.budgets[self.current_bidder] + self.members_forwards_needed[self.current_bidder] + self.member_defense_needed[self.current_bidder] + self.member_goalies_needed[self.current_bidder] < self.current_bid + 1:
                self.bidders_left[self.current_bidder] = 0
                self.current_bid -= 1
                #self.current_bidder = (self.current_bidder + 1) % self.num_players
            #check two - make sure we have roster space for this athlete
            elif self.nominated_player.position == self.Position.FORWARD:
                if self.members_forwards_needed[self.current_bidder] == 0:
                    self.bidders_left[self.current_bidder] = 0
                    self.current_bid -= 1
                    #self.current_bidder = (self.current_bidder + 1) % self.num_players
                    #return
            elif self.nominated_player.position == self.Position.DEFENSEMAN:
                if self.member_defense_needed[self.current_bidder] == 0:
                    self.bidders_left[self.current_bidder] = 0
                    self.current_bid -= 1
                    #self.current_bidder = (self.current_bidder + 1) % self.num_players
                    #return
            else:
                if self.member_goalies_needed[self.current_bidder] == 0:
                    self.bidders_left[self.current_bidder] = 0
                    self.current_bid -= 1
                    #self.current_bidder = (self.current_bidder + 1) % self.num_players
                    #return
                
            #now update bid and stuff

            if np.sum(self.bidders_left) <= 1:
                # someone else has won the player
                if np.sum(self.bidders_left) == 1:
                    winner = np.where(self.bidders_left == 1)[0][0]
                else:
                    winner = self.current_bidder
                self.budgets[winner] -= self.current_bid
                self.history[self.nominated_player] = { "price" : self.current_bid, "winner" : winner }
                self.current_bid = 0
                self.nominated_player.set_owner(winner)

                

                self.members_means[winner] += self.nominated_player.mean
                self.members_variances[winner] += self.nominated_player.variance

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
                    self.nominated_player = self.athletes.pop()
                    self.bidders_left = np.ones(self.num_players)
            self.current_bidder = (self.current_bidder + 1) % self.num_players
            while self.bidders_left[self.current_bidder] == 0:
                self.current_bidder = (self.current_bidder + 1) % self.num_players
                
            
        

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
        state = {
            "members_means": self.members_means,
            "members_variances": self.members_variances,
            "budgets": self.budgets,
            "members_forwards_needed": self.members_forwards_needed,
            "member_defense_needed": self.member_defense_needed,
            "member_goalies_needed": self.member_goalies_needed,
            "forwards_left": self.forwards_left,
            "defense_left": self.defense_left,
            "goalies_left": self.goalies_left,
            "nominated_player": {
                "mean" : self.nominated_player.mean,
                "variance" : self.nominated_player.variance,
                "position" : self.nominated_player.position
            },
            "current_bid": self.current_bid,
            "current_bidder": np.eye(self.num_players)[self.current_bidder],
            "bidders_left": self.bidders_left,
            "athletes_left" : {
                "forwards": sorted(
                [(athlete.mean, athlete.variance) for athlete in self.athletes if athlete.position == self.Position.FORWARD],
                key=lambda x: x[0],
                reverse=True
                ),
                "defensemen": sorted(
                [(athlete.mean, athlete.variance) for athlete in self.athletes if athlete.position == self.Position.DEFENSEMAN],
                key=lambda x: x[0],
                reverse=True
                ),
                "goalies": sorted(
                [(athlete.mean, athlete.variance) for athlete in self.athletes if athlete.position == self.Position.GOALIE],
                key=lambda x: x[0],
                reverse=True
                )
            }
        }

        return state
    
    def get_game_score(self):
        if self.game_over:
            outcomes = np.random.randn(NUM_SIMS, self.num_players) * np.sqrt(self.members_variances) + self.members_means
            ranks = outcomes.argsort(axis=1).argsort(axis=1) + 1
            return ranks.mean(axis=0)
        else:
            raise ValueError("Game is not over")


def generate_athletes(num_players, num_forwards, num_defensemen, num_goalies, shape, scale):
    """
    Generates a list of athletes with random mean and variance for each position type
    """
    athletes = []
    for _ in range(num_forwards * num_players):
        mean = gamma.rvs(a=shape, scale=scale)
        variance = 0.1 * mean + np.random.normal(0, 2)**2
        athletes.append(AuctionEnv.Athlete(mean=mean, variance=variance, position=AuctionEnv.Position.FORWARD))
    for _ in range(num_defensemen * num_players):
        mean = gamma.rvs(a=shape, scale=scale*0.55)
        variance = 0.1 * mean + np.random.normal(0, 2)**2
        athletes.append(AuctionEnv.Athlete(mean=mean, variance=variance, position=AuctionEnv.Position.DEFENSEMAN))
    for _ in range(num_goalies * num_players):
        athletes.append(AuctionEnv.Athlete(mean=gamma.rvs(a=shape, scale=scale), variance=gamma.rvs(a=shape, scale=scale), position=AuctionEnv.Position.GOALIE))
    return athletes

'''
athletes =  generate_athletes(8, 12, 6, 2, 2.5, 20)

for x in sorted(athletes, reverse=True):
    print(x)
'''