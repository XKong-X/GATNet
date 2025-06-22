import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque
import math
from torch import optim

class SparseLinear(nn.Module):
    def __init__(self, input_dim, output_dim, sparsity=0.2):
        super().__init__()
        self.register_buffer('mask', (torch.rand(output_dim, input_dim) > sparsity).float())
        self.weight = nn.Parameter(torch.Tensor(output_dim, input_dim))
        self.bias = nn.Parameter(torch.Tensor(output_dim))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
        self.weight.data *= self.mask

    def forward(self, x):
        return F.linear(x, self.weight * self.mask, self.bias)

class KalmanFilter(nn.Module):
    def __init__(self, state_dim):
        super().__init__()
        self.state_dim = state_dim
        self.register_buffer('F', torch.ones(state_dim))
        self.register_buffer('H', torch.ones(state_dim))
        self.register_buffer('Q', torch.ones(state_dim) * 0.01)
        self.register_buffer('R', torch.ones(state_dim) * 0.1)
        self.register_buffer('x_hat', torch.zeros(state_dim))
        self.register_buffer('P', torch.ones(state_dim) * 10.0)

    def predict(self):
        self.x_hat = self.F * self.x_hat
        self.P = self.F * self.P * self.F + self.Q

    def update(self, z):
        y = z - self.H * self.x_hat
        S = self.H * self.P * self.H + self.R
        if not torch.all(torch.isfinite(S)):
            return
        K = (self.P * self.H) / S
        self.x_hat += K * y
        self.P = self.P * (1 - K * self.H)

class EMA(nn.Module):
    def __init__(self, decay=0.95):
        super().__init__()
        self.decay = decay
        self.shadow = None

    def update(self, x):
        if self.shadow is None:
            self.shadow = x.clone().detach().to(x.device)
        else:
            self.shadow = self.decay * self.shadow + (1 - self.decay) * x

    def forward(self, x):
        self.update(x)
        return self.shadow

class ASP(nn.Module):
    def __init__(self, input_dim, embed_dim, sparsity=0.2, ema_decay=0.95):
        super().__init__()
        self.sparse_linear = SparseLinear(input_dim, embed_dim, sparsity)
        self.kalman = KalmanFilter(embed_dim)
        self.ema = EMA(ema_decay)
        self.kalman_map = nn.Linear(embed_dim, input_dim, bias=False)

    def forward(self, x, update_kalman=True):
        sparse_output = self.sparse_linear(x)
        if update_kalman:
            kalman_input = sparse_output.mean(dim=0)
            self.kalman.predict()
            self.kalman.update(kalman_input)
            updated_weight = self.kalman_map(self.kalman.x_hat).view(1, -1).repeat(self.sparse_linear.weight.size(0), 1)
            ema_weight = self.ema(updated_weight)
            with torch.no_grad():
                self.sparse_linear.weight.copy_(ema_weight)
        return sparse_output