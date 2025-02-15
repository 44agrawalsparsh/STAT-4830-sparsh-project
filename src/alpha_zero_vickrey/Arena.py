import logging
import itertools
from tqdm import tqdm
import random
import json
log = logging.getLogger(__name__)


class Arena():
    """
    An Arena class where any 2 agents can be pit against each other.
    """

    def __init__(self, players, game, display=None):
        """
        Input:
            players: list of functions that takes board as input, return action
            game: Game object
            display: a function that takes board as input and prints it (e.g.
                     display in othello/OthelloGame). Is necessary for verbose
                     mode.

        see othello/OthelloPlayers.py for an example. See pit.py for pitting
        human players/other baselines with each other.
        """

        assert game.getStructure()["NUM_PLAYERS"] == len(players)
        self.players = players
        self.game = game
        self.display = display

    def playGame(self, players, verbose=False):
        """
        Executes one episode of a game.

        Returns:
            scores
        """
        state = self.game.getInitState()
        print(json.dumps(state["athletes_starting"], indent=4))
        it = 0

        while type(self.game.getGameEnded(state)) is int:
            it += 1
            '''
            if verbose:
                assert self.display
                print("Turn ", str(it), "Player ", str(curPlayer))
                self.display(board)
            '''
            cur_player = self.game.getCurrentPlayer(state)
            action = players[cur_player](state)
            print(cur_player, action, state["nominated_player"], state["budgets"])

            '''valids = self.game.getValidMoves(state)

            if valids[action] == 0:
                log.error(f'Action {action} is not valid! Found on turn {it}')
                log.debug(f'valids = {valids}')
                assert valids[action] > 0'''

            '''
            # Notifying the opponent for the move
            opponent = players[-curPlayer + 1]
            if hasattr(opponent, "notify"):
                opponent.notify(board, action)
            '''

            state = self.game.getNextState(state, action)

        '''
        for player in players[0], players[2]:
            if hasattr(player, "endGame"):
                player.endGame()

        if verbose:
            assert self.display
            print("Game over: Turn ", str(it), "Result ", str(self.game.getGameEnded(board, 1)))
            self.display(board)
        '''
        return self.game.getGameEnded(state)

    def playGames(self, num, verbose=False):
        """
        Plays num games in which player1 starts num/2 games and player2 starts
        num/2 games.

        Returns:
            oneWon: games won by player1
            twoWon: games won by player2
            draws:  games won by nobody
        """

        original_scores = {i : 0 for i in range(len(self.players))}

        #perms = list(itertools.permutations([i for i in range(len(self.players))]))

        for _ in tqdm(range(num), desc="Games Time"):
            #perm = random.choice(perms)
            perm = range(len(self.players))
            players = [self.players[i] for i in perm]

            result = self.playGame(players)
            print(result)

            for i, score in enumerate(result):
                idx = perm[i]
                original_scores[idx] += score
                #print(original_scores)

        return original_scores
