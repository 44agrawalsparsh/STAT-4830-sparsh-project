import numpy as np
from basic_auction import BasicAuctionGame

game = BasicAuctionGame()

state = game.getInitState()
while type(game.getGameEnded(state)) is int:
    action = np.round(np.random.random())
    state = game.getNextState(state, action) 


state = game.getCanonicalForm(state, game.getCurrentPlayer(state))

symmetries = game.getSymmetries(state, [])


print(game.stringRepresentation(state))