# standard imports
from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time
import random
import math
import argparse
# local imports
from model import VRAM
from data_loader import VideoDataset
from torch.utils.data import DataLoader

# gpu settings 
use_cuda = torch.cuda.is_available()
print('gpu status ===',use_cuda)
torch.cuda.manual_seed(0)
device = torch.device("cuda" if use_cuda else "cpu")

def init(B, T, k):
    # T is the time frames usually 30 and 
    # k is the number of glimpses
    h_t = torch.randn(B, 1024).to(device)
    l_t = 2*(k/T) - 1.0
    return h_t, l_t

def train(model, device, optimiser, epochs, train_loader, val_loader):
    m = 10
    k = 3
    ng = 3
    for e in range(epochs):
        for i,sample in enumerate(train_loader):
            x = sample[0].to(device)
            y = sample[1].to(device)
            x = x.repeat(m,1,1,1,1)
            B,C,T,H,W = x.shape
            h_t, l_t = init(B, T, k)

            # extract the glimpses
            locs,log_pi,baselines = [],[],[]

            for t in range(ng - 1):
                # forward pass through model
                h_t, l_t, b_t, p = model(x, l_t, h_t)
                locs.append(l_t), baselines.append(b_t), log_pi.append(p)
            # last iteration
            h_t, l_t, b_t, log_probas, p = model(x, l_t, h_t, last=True)
            locs.append(l_t), baselines.append(b_t), log_pi.append(p)

            # convert list to tensors and reshape
            baselines = torch.stack(baselines).transpose(1, 0)
            log_pi = torch.stack(log_pi).transpose(1, 0)

            # average
            log_probas = log_probas.view(m, -1, log_probas.shape[-1])
            log_probas = torch.mean(log_probas, dim=0)

            baselines = baselines.contiguous().view(m, -1, baselines.shape[-1])
            baselines = torch.mean(baselines, dim=0)

            log_pi = log_pi.contiguous().view(m, -1, log_pi.shape[-1])
            log_pi = torch.mean(log_pi, dim=0)

            # calculate reward
            predicted = torch.max(log_probas, 1)[1]
            R = (predicted.detach() == y).float()
            R = R.unsqueeze(1).repeat(1, ng)

            # compute losses for differentiable modules
            loss_action = F.nll_loss(log_probas, y)
            loss_baseline = F.mse_loss(baselines, R)

            # compute reinforce loss
            # summed over timesteps and averaged across batch
            adjusted_reward = R - baselines.detach()
            loss_reinforce = torch.sum(-log_pi*adjusted_reward, dim=1)
            loss_reinforce = torch.mean(loss_reinforce, dim=0)

            # sum up into a hybrid loss
            loss = loss_action + loss_baseline + loss_reinforce
            
            reinforce_loss_total = reinforce_loss_total + loss_reinforce
            action_loss_total = action_loss_total + loss_action
            baseline_loss_total = baseline_loss_total + loss_baseline

            # compute accuracy
            correct = (predicted == y).float()
            acc = 100 * (correct.sum() / len(y))

            # store
            losses.update(loss.item(), x.size()[0])
            accs.update(acc.item(), x.size()[0])

            # compute gradients and update SGD
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # measure elapsed time
            toc = time.time()
            batch_time.update(toc-tic)
            # print('time taken for video',i,' ===',toc-tic1)
            pbar.set_description(
                (
                    "{:.1f}s - loss: {:.3f} - acc: {:.3f}".format(
                        (toc-tic), loss.item(), acc.item()
                    )
                )
            )
            pbar.update(self.batch_size)

        global loss_list, reinforce_list, baseline_list, action_list, acc_list

        reinforce_list.append(reinforce_loss_total/len(data_list))
        baseline_list.append(baseline_loss_total/len(data_list))
        action_list.append(action_loss_total/len(data_list))

        return losses.avg, accs.avg

# def validate(model, device, test_loader):
    
def main():
    # Training settings
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10,help='nepochs')
    parser.add_argument("--LR", type=float, default=1e-3,help='learning rate')
    parser.add_argument("--batch", type=float, default=1e-3,help='learning rate')

    args = parser.parse_args()

    nepochs = args.epochs
    LR = args.LR
    model = VRAM().to(device)
    optimiser = optim.Adam(model.parameters(), lr=LR)

    train_data = VideoDataset(dataset='ucf101', split='train', clip_len=30, preprocess=False)
    val_data = VideoDataset(dataset='ucf101', split='val', clip_len=30, preprocess=False)
    train_loader = DataLoader(train_data, batch_size=32, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_data, batch_size=32, shuflle=True, num_workers=4)

    train(model, device, optimiser, nepochs, train_loader, val_loader)

if __name__ == '__main__':
    main()