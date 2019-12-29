import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2

# gpu settings 
use_cuda = torch.cuda.is_available()
print('gpu status ===',use_cuda)
torch.cuda.manual_seed(0)
device = torch.device("cuda" if use_cuda else "cpu")

class retina(object):
    """
    inputs 
    - x: a 5D tensor of shape (B, T, C, H, W). The minibatch
      of images.
    - t: a 2D tensor of shape (B, 1). Contains normalized
      timestep in the range [-1, 1].
    - k: number of glimpse patches.
    - s: skipping factor that controls the size of
      successive patches.
      ****s is 1 by default****
    Returns
    -------
    - phi: a 5D tensor of shape (B, k*2, H, W).
    """
    def __init__(self, k, s):
        self.k = k
        self.s = s

    def foveate(self, x, l):
        """
        extract flow vectors from video frames g indicates the 
        number of flow vectors to be taken
        l is the time frame of the location vector
        k no of patches 
        s skipping factor

        The `k` patches are finally resized to (g*2, ) and
        concatenated into a tensor of shape (B, 1, k*2, H, W).
        """
        phi = []

        # extract 2*k patches from the loc list
        B, T, H, W, C = x.shape        
        lv = self.denormalize(T,l)
        loc = torch.arange(lv-self.k*self.s, lv+self.k*self.s+1, step=self.s)
        for i in loc:
            if(i != lv):
                phi.append(self.extract_patch(x, lv, i))

        # concatenate into a single tensor and flatten
        phi = torch.cat(phi, dim=1)
        return phi.unsqueeze(1)

    def extract_patch(self, x, a, b):
        """
        Extract a single set of flow vectors for the given x vector.
        between indices a,b
        Args
        ----
        - x: a 5D Tensor of shape (B, T, C, H, W). The minibatch
          of images.
        - a: a 2D Tensor of shape (B, 1).
        - b: a 2D Tensor of shape (B, 1).

        Returns
        -------
        - patch: a 5D Tensor of shape (B, 1, H, W)
        """
        # loop through mini-batch and extract
        B, T, H, W, C = x.shape        
        patch = []
        for i in range(B):
            im = x[i]
            a,b = self.exceed(a,b,T)

            leftframe = (im[0,a,:,:,:]).cpu()
            rightframe = (im[0,b,:,:,:]).cpu()

            leftframe = leftframe.numpy()
            rightframe = rightframe.numpy()

            leftgray = cv2.cvtColor(leftframe, cv2.COLOR_RGB2GRAY)
            rightgray = cv2.cvtColor(leftframe, cv2.COLOR_RGB2GRAY)

            # compute the flow
            flow = cv2.calcOpticalFlowFarneback(leftgray, rightgray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            flow = torch.tensor(flow).to(device)
            patch.append(flow)

        # concatenate into a single tensor
        patch = torch.stack(patch,dim=0)
        return patch

    def denormalize(self, T, coords):
        """
        Convert coordinates in the range [-1, 1] to
        coordinates in the range [0, T] where `T` is
        the length of the video sequence.
        """
        return (0.5 * ((coords + 1.0) * T)).long()

    def exceed(self, a, b, T):
        if(a<0):
            a = 0
        if(b>T):
            b = T-1
        return a,b

class glimpse_network(nn.Module):
    """
        `g_t = relu( fc( fc(l) ) + fc( fc(phi) ) )`

    Args
    ----
    - h_g: hidden layer size of the fc layer for `phi`.
    - h_l: hidden layer size of the fc layer for `l`.
    - k: number of patches to extract per glimpse.
    - s: skip factor.
    - x: a 5D Tensor of shape (B, 1, D, H, W). The minibatch
      of images.
    - l_t_prev: a 2D tensor of shape (B, 1). Contains the time
      frame t.

    Returns
    -------
    - g_t: a 2D tensor of shape (B, hidden_size). The glimpse
      representation returned by the glimpse network for the
      current timestep `t`.
    """
    def __init__(self, h_g, h_l, k, s, H, W):
        super(glimpse_network, self).__init__()
        self.retina = retina(k, s)

        # glimpse layer
        self.conv1 = nn.Conv3d()
        Din = 33*21
        self.fc1 = nn.Linear(Din, 256)
        self.fhc1 = nn.Linear(256, h_g)

        # location layer
        D_in = 1
        self.fc2 = nn.Linear(D_in, h_l)

        self.fc3 = nn.Linear(h_g, h_g+h_l)
        self.fc4 = nn.Linear(h_l, h_g+h_l)

        self.bn1 = nn.BatchNorm1d(h_g)
        self.bn2 = nn.BatchNorm1d(h_l)

    def forward(self, x, l_t_prev):
        # generate glimpse phi from image x
        phi = self.retina.foveate(x, l_t_prev)
        phi = self.conv1(phi)
        phi = phi.view(phi.shape[0],-1)
        
        
        # flatten location vector
        l_t_prev = l_t_prev.view(l_t_prev.size(0), -1)

        # feed phi and l to respective fc layers
        phi_out = F.relu(self.bn1(self.fhc1(self.fc1(phi))))
        l_out = F.relu(self.bn2(self.fc2(l_t_prev)))

        # feed to fc layer
        g_t = F.relu(self.fc3(phi_out) + self.fc4(l_out))

        return g_t

class core_network(nn.Module):
    """
        `h_t = relu( fc(h_t_prev) + fc(g_t) )`

    Args
    ----
    - input_size: input size of the rnn.
    - hidden_size: hidden size of the rnn.
    - g_t: a 2D tensor of shape (B, hidden_size). The glimpse
      representation returned by the glimpse network for the
      current timestep `t`.
    - h_t_prev: a 2D tensor of shape (B, hidden_size). The
      hidden state vector for the previous timestep `t-1`.

    Returns
    -------
    - h_t: a 2D tensor of shape (B, hidden_size). The hidden
      state vector for the current timestep `t`.
    """
    def __init__(self, input_size, hidden_size):
        super(core_network, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.i2h = nn.Linear(input_size, hidden_size)
        self.h2h = nn.Linear(hidden_size, hidden_size)

    def forward(self, g_t, h_t_prev):
        h1 = self.i2h(g_t)
        h2 = self.h2h(h_t_prev)
        h_t = F.relu(h1 + h2)
        return h_t

class action_network(nn.Module):
    """
    Args
    ----
    - input_size: input size of the fc layer.
    - output_size: output size of the fc layer.
    - h_t: the hidden state vector of the core network for
      the current time step `t`.

    Returns
    -------
    - a_t: output probability vector over the classes.
    """
    def __init__(self, input_size, output_size):
        super(action_network, self).__init__()
        self.fc = nn.Linear(input_size, output_size)

    def forward(self, h_t):
        a_t = F.log_softmax(self.fc(h_t), dim=1)
        return a_t

class location_network(nn.Module):
    """
    Args
    ----
    - input_size: input size of the fc layer.
    - output_size: output size of the fc layer.
    - std: standard deviation of the normal distribution.
    - h_t: the hidden state vector of the core network for
      the current time step `t`.

    Returns
    -------
    - mu: a 2D vector of shape (B, 1).
    - l_t: a 2D vector of shape (B, 1).
    """
    def __init__(self, input_size, output_size, std):
        super(location_network, self).__init__()
        self.std = std
        self.fc = nn.Linear(input_size, output_size)

    def forward(self, h_t):
        # compute mean
        mu = torch.tanh(self.fc(h_t.detach()))

        # reparametrization trick
        noise = torch.zeros_like(mu)
        noise.data.normal_(std=self.std)
        l_t = mu + noise

        # bound between [-1, 1]
        l_t = torch.tanh(l_t)

        return mu, l_t

class baseline_network(nn.Module):
    """
    Regresses the baseline in the reward function
    to reduce the variance of the gradient update.

    Args
    ----
    - input_size: input size of the fc layer.
    - output_size: output size of the fc layer.
    - h_t: the hidden state vector of the core network
      for the current time step `t`.

    Returns
    -------
    - b_t: a 2D vector of shape (B, 1). The baseline
      for the current time step `t`.
    """
    def __init__(self, input_size, output_size):
        super(baseline_network, self).__init__()
        self.fc = nn.Linear(input_size, output_size)

    def forward(self, h_t):
        b_t = F.relu(self.fc(h_t.detach()))
        return b_t
