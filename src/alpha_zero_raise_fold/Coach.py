import logging
import os
import sys
from collections import deque
from pickle import Pickler, Unpickler
from random import shuffle

import numpy as np
from tqdm import tqdm

from Arena import Arena #TODO
from MCTS import MCTS, VanillaMCTS #TODO
import copy
log = logging.getLogger(__name__)


class Coach():
    """
    This class executes the self-play + learning. It uses the functions defined
    in Game and NeuralNet. args are specified in main.py.
    """

    def __init__(self, game, nnet, args):
        self.game = game
        self.nnet = nnet
        self.pnet = self.nnet.__class__(self.game)  # the competitor network
        self.args = args
        self.mcts = MCTS(self.game, self.nnet, self.args)
        self.vmcts = VanillaMCTS(self.game, self.args.numVMCTSSims)
        self.trainExamplesHistory = []  # history of examples from args.numItersForTrainExamplesHistory latest iterations
        self.skipFirstSelfPlay = False  # can be overriden in loadTrainExamples()


    def executeVanillaEpisode(self):
        """ 
        
        To "warm" start the neural network we derive some training examples from an end to end raw monte carlo. Where our target policy is the move a vanilla monte carlo method would chosoe at each step with the reward vector being said move's reward.
        
        """
        trainExamples = []
        state = self.game.getInitState()
        self.curPlayer = 0 #0 is our player
        episodeStep = 0

        while True:
            episodeStep += 1
            print("Episode Step", episodeStep)

            self.curPlayer = self.game.getCurrentPlayer(state)
            pi, v = self.vmcts.vanilla_monte_carlo(state)

            new_v = copy.deepcopy(v)
            true_val = v[self.curPlayer]
            other_val = v[0]

            new_v[self.curPlayer] = other_val
            new_v[0] = true_val
            trainExamples.append((state, pi, new_v))
            print(pi, new_v)

            action = np.random.choice(len(pi), p=pi)
            state = self.game.getNextState(state, action)

            r = self.game.getGameEnded(state)

            if type(r) is not int:
                return trainExamples
            

    def executeEpisode(self):
        """
        This function executes one episode of self-play, starting with player 1.
        As the game is played, each turn is added as a training example to
        trainExamples. The game is played till the game ends. After the game
        ends, the outcome of the game is used to assign values to each example
        in trainExamples.

        It uses a temp=1 if episodeStep < tempThreshold, and thereafter
        uses temp=0.

        Returns:
            trainExamples: a list of examples of the form (canonicalBoard, currPlayer, pi,v)
                           pi is the MCTS informed policy vector, v is +1 if
                           the player eventually won the game, else -1.
        """
        trainExamples = []
        state = self.game.getInitState()
        self.curPlayer = 0 #0 is our player
        episodeStep = 0

        while True:
            episodeStep += 1
            print("Episode Step", episodeStep)
            #canonicalBoard = self.game.getCanonicalForm(state, self.curPlayer)
            temp = int(episodeStep < self.args.tempThreshold)

            pi = self.mcts.getActionProb(state, temp=temp)
            print(pi)
            sym = [(state, pi)]#self.game.getSymmetries(state, pi, self.curPlayer) 

            ## ^^^ might do some other data augmentation later not sure

            #pi = self.mcts.getActionProb(canonicalBoard, temp=temp)
            #sym = self.game.getSymmetries(canonicalBoard, pi)
            for s, p in sym:
                trainExamples.append([s, self.curPlayer, p, None])
                ###need to keep the current player here so we can understand the end value of this state

            action = np.random.choice(len(pi), p=pi)
            state = self.game.getNextState(state, action)
            self.curPlayer = self.game.getCurrentPlayer(state)

            r = self.game.getGameEnded(state)

            if type(r) is not int:
                print("Adding Examples")
                
                output = []
                for x in trainExamples:

                    new_r = copy.deepcopy(r)
                    
                    true_val = r[x[1]]
                    other_val = r[0]

                    new_r[x[1]] = other_val
                    new_r[0] = true_val

                    output.append((x[0], x[2], new_r))

                    print(x[2], new_r)
                return output

    def learn(self):
        """
        Performs numIters iterations with numEps episodes of self-play in each
        iteration. After every iteration, it retrains neural network with
        examples in trainExamples (which has a maximum length of maxlenofQueue).
        It then pits the new neural network against the old one and accepts it
        only if it wins >= updateThreshold fraction of games.
        """

        '''

        def flip(x):
            valid_moves = self.game.getValidMoves(x)
            moves = [i for i,x in enumerate(valid_moves) if x == 1]
            return np.random.choice(moves)
        
        def vmc(state):
            pi, _ = self.vmcts.vanilla_monte_carlo(state)
            action = np.random.choice(len(pi), p=pi)
            print(action)
            return action

        num_players = self.game.getStructure()["NUM_PLAYERS"]
        players = [flip for i in range(num_players - 1)]
        players.append(vmc)
        arena = Arena(players, self.game)
        results = arena.playGames(self.args.arenaCompare)
        print(results)


        '''

        warm_examples = []
        for _ in tqdm(range(self.args.warmGames), desc="Warm Up"):
            warm_examples.extend(self.executeVanillaEpisode())
        
        shuffle(warm_examples)


        # training new network, keeping a copy of the old one
        self.nnet.save_checkpoint(folder=self.args.checkpoint, filename='temp.pth.tar')
        self.pnet.load_checkpoint(folder=self.args.checkpoint, filename='temp.pth.tar')
        
        num_players = self.game.getStructure()["NUM_PLAYERS"]
        num_old = num_players // 2
        num_new = num_players - num_old

        self.nnet.train(warm_examples)

        log.info('PITTING AGAINST PREVIOUS VERSION')
        players = []
        for i in range(num_old):
            players.append(MCTS(self.game, self.pnet, self.args))
        for i in range(num_old, num_old + num_new):
            players.append(MCTS(self.game, self.nnet, self.args))

        players = [(lambda x : np.argmax(p.getActionProb(x, temp=0))) for p in players]
        arena = Arena(players, self.game)

        results = arena.playGames(self.args.arenaCompare)

        old_record = 0
        new_record = 0
        for i in range(num_old):
            old_record += results[i] / num_old
        for i in range(num_old, num_old + num_new):
            new_record += results[i] / num_new

        log.info(f"Old Record: {old_record}, New Record : {new_record}")

        if new_record / (new_record + old_record) < self.args.updateThreshold:
            log.info('REJECTING NEW MODEL')
            self.nnet.load_checkpoint(folder=self.args.checkpoint, filename='temp.pth.tar')
        else:
            log.info('ACCEPTING NEW MODEL')
            self.nnet.save_checkpoint(folder=self.args.checkpoint, filename=self.getCheckpointFile(i))
            self.nnet.save_checkpoint(folder=self.args.checkpoint, filename='best.pth.tar')

        for i in range(2, self.args.numIters + 2):
            # bookkeeping
            log.info(f'Starting Iter #{i} ...')
            # examples of the iteration
            if not self.skipFirstSelfPlay or i > 1:
                iterationTrainExamples = deque([], maxlen=self.args.maxlenOfQueue)

                for _ in tqdm(range(self.args.numEps), desc="Self Play"):
                    self.mcts = MCTS(self.game, self.nnet, self.args)  # reset search tree
                    iterationTrainExamples += self.executeEpisode()

                # save the iteration examples to the history 
                self.trainExamplesHistory.append(iterationTrainExamples)

            if len(self.trainExamplesHistory) > self.args.numItersForTrainExamplesHistory:
                log.warning(
                    f"Removing the oldest entry in trainExamples. len(trainExamplesHistory) = {len(self.trainExamplesHistory)}")
                self.trainExamplesHistory.pop(0)
            # backup history to a file
            # NB! the examples were collected using the model from the previous iteration, so (i-1)  
            self.saveTrainExamples(i - 1)

            # shuffle examples before training
            trainExamples = []
            for e in self.trainExamplesHistory:
                trainExamples.extend(e)
            shuffle(trainExamples)

            # training new network, keeping a copy of the old one
            self.nnet.save_checkpoint(folder=self.args.checkpoint, filename='temp.pth.tar')
            self.pnet.load_checkpoint(folder=self.args.checkpoint, filename='temp.pth.tar')
            
            num_players = self.game.getStructure()["NUM_PLAYERS"]
            num_old = num_players // 2
            num_new = num_players - num_old

            self.nnet.train(trainExamples)

            log.info('PITTING AGAINST PREVIOUS VERSION')
            players = []
            for i in range(num_old):
                players.append(MCTS(self.game, self.pnet, self.args))
            for i in range(num_old, num_old + num_new):
                players.append(MCTS(self.game, self.nnet, self.args))

            players = [(lambda x : np.argmax(p.getActionProb(x, temp=0))) for p in players]
            arena = Arena(players, self.game)

            results = arena.playGames(self.args.arenaCompare)

            old_record = 0
            new_record = 0
            for i in range(num_old):
                old_record += results[i] / num_old
            for i in range(num_old, num_old + num_new):
                new_record += results[i] / num_new

            log.info(f"Old Record: {old_record}, New Record : {new_record}")

            if new_record / (new_record + old_record) < self.args.updateThreshold:
                log.info('REJECTING NEW MODEL')
                self.nnet.load_checkpoint(folder=self.args.checkpoint, filename='temp.pth.tar')
            else:
                log.info('ACCEPTING NEW MODEL')
                self.nnet.save_checkpoint(folder=self.args.checkpoint, filename=self.getCheckpointFile(i))
                self.nnet.save_checkpoint(folder=self.args.checkpoint, filename='best.pth.tar')
        

    def getCheckpointFile(self, iteration):
        return 'checkpoint_' + str(iteration) + '.pth.tar'

    def saveTrainExamples(self, iteration):
        folder = self.args.checkpoint
        if not os.path.exists(folder):
            os.makedirs(folder)
        filename = os.path.join(folder, self.getCheckpointFile(iteration) + ".examples")
        with open(filename, "wb+") as f:
            Pickler(f).dump(self.trainExamplesHistory)
        f.closed

    def loadTrainExamples(self):
        modelFile = os.path.join(self.args.load_folder_file[0], self.args.load_folder_file[1])
        examplesFile = modelFile + ".examples"
        if not os.path.isfile(examplesFile):
            log.warning(f'File "{examplesFile}" with trainExamples not found!')
            r = input("Continue? [y|n]")
            if r != "y":
                sys.exit()
        else:
            log.info("File with trainExamples found. Loading it...")
            with open(examplesFile, "rb") as f:
                self.trainExamplesHistory = Unpickler(f).load()
            log.info('Loading done!')

            # examples based on the model were already collected (loaded)
            self.skipFirstSelfPlay = True
