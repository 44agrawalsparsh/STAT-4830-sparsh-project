import gymnasium as gym
from gymnasium import spaces
import numpy as np
from auction_env import AuctionEnv, generate_athletes

class AuctionGymEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        super(AuctionGymEnv, self).__init__()
        self.num_players = 6
        self.num_forwards = 8
        self.num_defensemen = 4
        self.num_goalies = 2
        self.shape = 2.5
        self.scale = 20
        self.budgets = 100
        self.athletes = generate_athletes(self.num_players, self.num_forwards, self.num_defensemen, self.num_goalies, self.shape, self.scale)
        self.env = AuctionEnv(self.athletes, self.num_players, self.budgets, self.num_forwards, self.num_defensemen, self.num_goalies)

        # Define action and observation space
        self.action_space = spaces.Discrete(2)  # Two actions: pass (0) or bid (1)
        self.observation_space = spaces.Dict({
            "members_means": spaces.Box(low=0, high=np.inf, shape=(self.num_players,), dtype=np.float32),
            "budgets": spaces.Box(low=0, high=np.inf, shape=(self.num_players,), dtype=np.int32),
            "members_forwards_needed": spaces.Box(low=0, high=self.num_forwards, shape=(self.num_players,), dtype=np.int32),
            "member_defense_needed": spaces.Box(low=0, high=self.num_defensemen, shape=(self.num_players,), dtype=np.int32),
            "member_goalies_needed": spaces.Box(low=0, high=self.num_goalies, shape=(self.num_players,), dtype=np.int32),
            "forwards_left": spaces.Box(low=0, high=self.num_players * self.num_forwards, shape=(), dtype=np.int32),
            "defense_left": spaces.Box(low=0, high=self.num_players * self.num_defensemen, shape=(), dtype=np.int32),
            "goalies_left": spaces.Box(low=0, high=self.num_players * self.num_goalies, shape=(), dtype=np.int32),
            "nominated_player": spaces.Dict({
                "mean": spaces.Box(low=0, high=np.inf, shape=(), dtype=np.float32),
                "position": spaces.Discrete(3)  # 0: Forward, 1: Defenseman, 2: Goalie
            }),
            "current_bid": spaces.Box(low=0, high=np.inf, shape=(), dtype=np.int32),
            "current_bidder": spaces.MultiBinary(self.num_players),
            "bidders_left": spaces.MultiBinary(self.num_players),
            "athletes_left": spaces.Dict({
                "forwards": spaces.Box(low=0, high=np.inf, shape=(self.num_players * self.num_forwards,), dtype=np.float32),
                "defensemen": spaces.Box(low=0, high=np.inf, shape=(self.num_players * self.num_defensemen,), dtype=np.float32),
                "goalies": spaces.Box(low=0, high=np.inf, shape=(self.num_players * self.num_goalies,), dtype=np.float32)
            })
        })

    def reset(self):
        self.athletes = generate_athletes(self.num_players, self.num_forwards, self.num_defensemen, self.num_goalies, self.shape, self.scale)
        self.env = AuctionEnv(self.athletes, self.num_players, self.budgets, self.num_forwards, self.num_defensemen, self.num_goalies)
        return self.env.get_state()

    def step(self, action):
        self.env.play_action(action)
        state = self.env.get_state()
        reward = 0  # Define your reward function here
        done = self.env.game_over
        info = {}
        return state, reward, done, info

    def render(self, mode='human'):
        state = self.env.get_state()
        #print(state)

    def get_payouts(self):
        return self.env.get_game_score()

    def close(self):
        pass

    def get_legal_actions(self):
        # Return a list of legal actions
        return self.env.get_legal_actions()

    def get_current_player(self):
        # Return the index of the current player
        return self.env.get_current_player()
