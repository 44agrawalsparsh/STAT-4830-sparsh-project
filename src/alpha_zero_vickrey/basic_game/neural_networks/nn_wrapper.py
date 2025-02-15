import os
import sys
import time

import numpy as np
from tqdm import tqdm

sys.path.append('../../')
from NeuralNet import NeuralNet
import torch
import torch.optim as optim

from .game_nn import AuctionEnvNet as aucnet
from utils import dotdict, AverageMeter
import torch.nn.functional as F

args = dotdict({
    'lr': 0.001,
    'dropout': 0.1,
    'epochs': 10,
    'batch_size': 64,
    'device' : torch.device(
        "cuda" if torch.cuda.is_available() else
        "mps" if torch.backends.mps.is_available() else
        "cpu"
    )
})


class NNetWrapper(NeuralNet):
    def __init__(self, game):
        self.nnet = aucnet(game)
        self.action_size = game.getActionSize()
        self.game = game

        self.nnet.to(args.device)


    def train(self, examples):
        """
        examples: list of examples, each example is of form (state, pi, v)
        """
        optimizer = optim.Adam(self.nnet.parameters(), lr=args.lr)

        for epoch in range(args.epochs):
            print('EPOCH ::: ' + str(epoch + 1))
            self.nnet.train()
            pi_losses = AverageMeter()
            v_losses = AverageMeter()

            batch_count = int(len(examples) / args.batch_size)

            t = tqdm(range(batch_count), desc='Training Net')
            for _ in t:
                sample_ids = np.random.randint(len(examples), size=args.batch_size)
                states, pis, vs = list(zip(*[examples[i] for i in sample_ids]))

                states = self.game.prepare_tensor_input(states, device=args.device)

                target_pis = torch.FloatTensor(np.array(pis)).to(args.device)
                vs_np = np.array(vs, dtype=np.float32)  # Ensure shape (N, 6)
                target_vs = torch.from_numpy(vs_np).to(args.device)
                # predict

                # compute output
                out_pi, out_v = self.nnet(states)
                l_pi = self.loss_pi(target_pis, out_pi)
                l_v = self.loss_v(target_vs, out_v)
                total_loss = l_pi + l_v

                # record loss
                batch_size = states["global"].size(0)
                pi_losses.update(l_pi.item(), batch_size)
                v_losses.update(l_v.item(), batch_size)
                t.set_postfix(Loss_pi=pi_losses, Loss_v=v_losses)

                # compute gradient and do SGD step
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()

    def predict(self, state):
        """
        state: Dictionary representing the current game state
        """
        start = time.time()

        #cur_bidder = self.game.getCurrentPlayer(state)

        #canonical = self.game.getCanonicalForm(state, cur_bidder)

        # Prepare input tensors from game state
        input_tensors = self.game.prepare_tensor_input([state], device=args.device)

        # Set model to evaluation mode
        self.nnet.eval()

        with torch.no_grad():
            pi, v = self.nnet(input_tensors)

        # Convert outputs to NumPy
        pi = torch.exp(pi).data.cpu().numpy()[0]  # Convert log probabilities back to probabilities
        v = v.data.cpu().numpy()[0]

        return pi, v


    def loss_pi(self, targets, outputs):
        return -torch.sum(targets * outputs) / targets.size()[0]

    def loss_v(self, targets, outputs):
        return F.mse_loss(outputs, targets)

    def save_checkpoint(self, folder='checkpoint', filename='checkpoint.pth.tar'):
        filepath = os.path.join(folder, filename)
        if not os.path.exists(folder):
            print("Checkpoint Directory does not exist! Making directory {}".format(folder))
            os.mkdir(folder)
        else:
            print("Checkpoint Directory exists! ")
        torch.save({
            'state_dict': self.nnet.state_dict(),
        }, filepath)

    def load_checkpoint(self, folder='checkpoint', filename='checkpoint.pth.tar'):
        # https://github.com/pytorch/examples/blob/master/imagenet/main.py#L98
        filepath = os.path.join(folder, filename)
        if not os.path.exists(filepath):
            raise ("No model in path {}".format(filepath))
        map_location = args.device
        checkpoint = torch.load(filepath, map_location=map_location)
        self.nnet.load_state_dict(checkpoint['state_dict'])
