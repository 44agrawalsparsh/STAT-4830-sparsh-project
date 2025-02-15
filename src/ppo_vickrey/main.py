from basic_game.neural_networks.ppo_nn import PPONNetWrapper
from basic_game.basic_auction import BasicAuctionGame


if __name__ == "__main__":
    game = BasicAuctionGame()
    ppo = PPONNetWrapper(game)
    ppo.learn()