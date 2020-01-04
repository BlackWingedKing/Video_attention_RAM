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
from data_loader_ucf101 import VideoDataset
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
import os

writer = SummaryWriter('tensorboard/gru_resnet_cbam')
# gpu settings 
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
use_cuda = torch.cuda.is_available()
print('gpu status ===',use_cuda)
torch.cuda.manual_seed(0)
device = torch.device("cuda" if use_cuda else "cpu")

def init(B, T, k):
    # T is the time frames usually 30 and 
    # k is the number of glimpses
    h_t = torch.randn(B, 1024).to(device)
    l_t = (2*(k/T) - 1.0)*torch.ones(B,1).to(device)
    return h_t, l_t

def train(model, device, optimiser, epochs, train_loader, val_loader):
    m = 10 # monte carlo sampling factor
    k = 0
    ng = 7
    for e in range(epochs):
        tic = time.time()
        r_loss,a_loss,b_loss,acc = 0.0,0.0,0.0,0.0
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

            baselines = baselines.view(m, -1, baselines.shape[-1])
            baselines = torch.mean(baselines, dim=0)

            log_pi = log_pi.view(m, -1, log_pi.shape[-1])
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
            
            r_loss+=loss_reinforce.item()
            a_loss+=loss_action.item()
            b_loss+=loss_baseline.item()

            # compute accuracy
            correct = (predicted == y).float()
            acc+= 100 * (correct.sum() / len(y))

            # compute gradients and update SGD
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
        writer.add_scalar('train_acc',acc/len(train_loader),e)
        writer.add_scalar('train_r loss',r_loss/len(train_loader),e)
        writer.add_scalar('train_a_loss',a_loss/len(train_loader),e)
        writer.add_scalar('train_b_loss',b_loss/len(train_loader),e)
        writer.add_scalar('train_loss',(r_loss+a_loss+b_loss)/len(train_loader),e)

        validate(model, device, val_loader, e)
        print('epoch:',e,' completed', 'time taken:',time.time() - tic)

def validate(model, device, val_loader,e):
    m = 1 # monte carlo sampling factor
    k = 0
    ng = 7
    with torch.no_grad():
        r_loss,a_loss,b_loss,acc = 0.0,0.0,0.0,0.0
        for i,sample in enumerate(val_loader):
            tic = time.time()
            x = sample[0].to(device)
            # print("x.shape",x.shape)
            y = sample[1].to(device)
            # print("y.shape",y.shape)
            # print("_____________________________")
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

            baselines = baselines.view(m, -1, baselines.shape[-1])
            baselines = torch.mean(baselines, dim=0)

            log_pi = log_pi.view(m, -1, log_pi.shape[-1])
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
            
            r_loss+=loss_reinforce.item()
            a_loss+=loss_action.item()
            b_loss+=loss_baseline.item()

            # compute accuracy
            correct = (predicted == y).float()
            acc+= 100 * (correct.sum() / len(y))

        writer.add_scalar('val_acc',acc/len(val_loader),e)
        writer.add_scalar('val_r loss',r_loss/len(val_loader),e)
        writer.add_scalar('val_a_loss',a_loss/len(val_loader),e)
        writer.add_scalar('val_b_loss',b_loss/len(val_loader),e)
        writer.add_scalar('val_loss',(r_loss+a_loss+b_loss)/len(val_loader),e)

def main():
    # Training settings
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=200,help='nepochs')
    parser.add_argument("--LR", type=float, default=1e-3,help='learning rate')
    parser.add_argument("--batch", type=float, default=64,help='learning rate')

    args = parser.parse_args()

    nepochs = args.epochs
    LR = args.LR
    model = VRAM().to(device)
    optimiser = optim.Adam(model.parameters(), lr=LR)

    train_data = VideoDataset(dataset='ucf101', split='train', clip_len=30, preprocess=False)
    val_data = VideoDataset(dataset='ucf101', split='val', clip_len=30, preprocess=False)
    train_loader = DataLoader(train_data, batch_size=64, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_data, batch_size=64, shuffle=True, num_workers=0)

    train(model, device, optimiser, nepochs, train_loader, val_loader)
    # validate(model, device, val_loader, 1)

    writer.export_scalars_to_json("metrics.json")
    writer.close()

if __name__ == '__main__':
    main()
