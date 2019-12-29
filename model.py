import math

import torch
import torch.nn as nn

from torch.distributions import Normal

from modules import baseline_network
from modules import glimpse_network, core_network
from modules import action_network, location_network

class VRAM(nn.Module):
    def __init__(self):
        super(VRAM, self).__init__()

        self.std = std
        self.sensor = glimpse_network(3)
        self.rnn = core_network()
        self.locator = location_network(std=0.2)
        self.classifier = action_network()
        self.baseliner = baseline_network()

    def forward(self, x, l_t_prev, h_t_prev, last=False):
        """
        Run the recurrent attention model for 1 timestep
        on the minibatch of images `x`.

        Args
        ----
        - x: a 5D Tensor of shape (B, T, H, W, C). The minibatch
          of images.
        - l_t_prev: a 2D tensor of shape (B, 1). The location vector
          containing the glimpse coordinates [x, y] for the previous
          timestep `t-1`. It is normalized
        - h_t_prev: a 2D tensor of shape (B, hidden_size). The hidden
          state vector for the previous timestep `t-1`.
        - last: a bool indicating whether this is the last timestep.
          If True, the action network returns an output probability
          vector over the classes and the baseline `b_t` for the
          current timestep `t`. Else, the core network returns the
          hidden state vector for the next timestep `t+1` and the
          location vector for the next timestep `t+1`.

        Returns
        -------
        - h_t: a 2D tensor of shape (B, hidden_size). The hidden
          state vector for the current timestep `t`.
        - mu: a 2D tensor of shape (B, 1). The mean that parametrizes
          the Gaussian policy.
        - l_t: a 2D tensor of shape (B, 1). The location vector
          containing the glimpse coordinates [x, y] for the
          current timestep `t`.
        - b_t: a vector of length (B,). The baseline for the
          current time step `t`.
        - log_probas: a 2D tensor of shape (B, num_classes). The
          output log probability vector over the classes.
        - log_pi: a vector of length (B,).
        """
        g_t = self.sensor(x, l_t_prev)
        h_t = self.rnn(g_t, h_t_prev)
        mu, l_t = self.locator(h_t)
        b_t = self.baseliner(h_t).squeeze(dim=1)
        # b_t = self.baseliner(h_t)
        # we assume both dimensions are independent
        # 1. pdf of the joint is the product of the pdfs
        # 2. log of the product is the sum of the logs
        log_pi = Normal(mu, self.std).log_prob(l_t)
        log_pi = torch.sum(log_pi, dim=1)

        if last:
            log_probas = self.classifier(h_t)
            return h_t, l_t, b_t, log_probas, log_pi

        return h_t, l_t, b_t, log_pi
