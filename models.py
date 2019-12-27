import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2

# gpu settings 
use_cuda = torch.cuda.is_available()
torch.cuda.manual_seed(config.random_seed)
device = torch.device("cuda" if use_cuda else "cpu")

class retina(object):
    """
    inputs 
    - x: a 5D tensor of shape (B, T, H, W, C). The minibatch
      of images.
    - t: a 2D tensor of shape (B, 1). Contains normalized
      timestep in the range [-1, 1].
    - g: no. of flow vectors in first glimpse patch.
      ****g is an even number by default 2****
    - k: number of glimpse patches.
    - s: skipping factor that controls the size of
      successive patches.
      ****s is 1 by default****
    Returns
    -------
    - phi: a 5D tensor of shape (B, k*g, H, W, 2).
    """
    def __init__(self, g, k, s):
        self.g = g
        self.k = k
        self.s = s

    def foveate(self, x, l):
        """
        extract flow vectors from video frames g indicates the 
        number of flow vectors to be taken
        l is the time frame of the location vector
        k no of patches 
        s skipping factor

        The `k` patches are finally resized to (g*k, ) and
        concatenated into a tensor of shape (B, k*g, H, W, C).
        """
        phi = []
        skip = self.s

        # extract k patches of increasing skip
        for i in range(self.k):
            # change here 
            phi.append(self.extract_patch(x, l, self.g, skip))
            skip = int(self.s + skip)

        # concatenate into a single tensor and flatten
        phi = torch.cat(phi, dim=1)
        return phi

    def extract_patch(self, x, l, g, skip):
        """
        Extract a single set of flow vectors for the given x vector.

        Args
        ----
        - x: a 5D Tensor of shape (B, T, H, W, C). The minibatch
          of images.
        - l: a 2D Tensor of shape (B, 1).
        - skip: a scalar defining the skip value.

        Returns
        -------
        - patch: a 5D Tensor of shape (B, g, H, W, 2)
        """
        B, T, H, W, C = x.shape

        # denormalize coords of patch center
        coords = self.denormalize(T, l)

        # compute top left corner of patch
        patch_t = coords[:, 0]
        
        # loop through mini-batch and extract
        patch = []
        for i in range(B):

            im = x[i].unsqueeze(dim=0)
            patch_t[i] = self.exceed(patch_t[i],g,skip,T)

            # compute slice indices
            from_t, to_t = patch_t[i] - skip*(g//2), patch_t[i] + skip*(g//2)

            # cast to ints
            from_t, to_t = from_t.item(), to_t.item()
            middle_t = (from_t + to_t)//2

            # now since we got these compute the optical flow
            # convert the numpy arrays to cv images

            leftframe = (im[0,from_t,:,:,:]).cpu()
            middleframe = (im[0,middle_t,:,:,:]).cpu()
            rightframe = (im[0,to_t,:,:,:]).cpu()

            leftframe = leftframe.numpy()
            middleframe = middleframe.numpy()
            rightframe = rightframe.numpy()

            leftframe = cv2.cvtColor(leftframe, cv2.COLOR_RGB2BGR)
            middleframe = cv2.cvtColor(middleframe, cv2.COLOR_RGB2BGR)
            rightframe = cv2.cvtColor(rightframe, cv2.COLOR_RGB2BGR)

            leftgray = cv2.cvtColor(leftframe, cv2.COLOR_BGR2GRAY)
            middlegray = cv2.cvtColor(leftframe, cv2.COLOR_BGR2GRAY)
            rightgray = cv2.cvtColor(leftframe, cv2.COLOR_BGR2GRAY)

            # compute the flow
            flow1 = cv2.calcOpticalFlowFarneback(leftgray, middlegray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            flow2 = cv2.calcOpticalFlowFarneback(middlegray, rightgray, None, 0.5, 3, 15, 3, 5, 1.2, 0)

            flow1 = np.asarray(flow1)
            flow2 = np.asarray(flow2)

            flow = np.stack((flow1,flow2),axis=0)

            flow = torch.tensor(flow)
            
            if(config.use_gpu):
              flow = flow.to(device)
            else:
              flow = flow.cpu()
            
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

    def exceed(self, patch_t, g, skip, T):
        left = patch_t - skip*(g//2)
        right = patch_t + skip*(g//2) 
        if (left < 0):
            a = skip*(g//2) + 1
            return a
        elif(right >= T):
            a = T-2-skip*(g//2)
            return a
        else:
          return patch_t



class glimpse_network(nn.Module):
    """
        `g_t = relu( fc( fc(l) ) + fc( fc(phi) ) )`

    Args
    ----
    - h_g: hidden layer size of the fc layer for `phi`.
    - h_l: hidden layer size of the fc layer for `l`.
    - g: size of the square patches in the glimpses extracted
      by the retina.
    - k: number of patches to extract per glimpse.
    - s: scaling factor that controls the size of successive patches.
    - c: number of channels in each image.
    - x: a 4D Tensor of shape (B, H, W, C). The minibatch
      of images.
    - l_t_prev: a 2D tensor of shape (B, 2). Contains the glimpse
      coordinates [x, y] for the previous timestep `t-1`.

    Returns
    -------
    - g_t: a 2D tensor of shape (B, hidden_size). The glimpse
      representation returned by the glimpse network for the
      current timestep `t`.
    """
    def __init__(self, h_g, h_l, g, k, s, c, H, W):
        super(glimpse_network, self).__init__()

        self.retina = retina(g, k, s)

        # glimpse layer
        self.conv1 = nn.Conv3d(g*k,1,(20,20,2),stride = (5,5,1))
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