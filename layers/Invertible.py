import torch
import torch.nn as nn
import torch.nn.functional as F

class RevIN(nn.Module):
    def __init__(self, num_features: int, eps=1e-5, affine=False):
        """
        :param num_features: the number of features or channels
        :param eps: a value added for numerical stability
        :param affine: if True, RevIN has learnable affine parameters
        """
        super(RevIN, self).__init__()

        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        
        if self.affine:
            self._init_params()

    def forward(self, x, mode:str):
        if mode == 'norm':
            self._get_statistics(x)
            x = self._normalize(x)
        
        elif mode == 'denorm':
            x = self._denormalize(x)
        
        else: raise NotImplementedError

        return x

    def _init_params(self):
        # initialize RevIN params: (C,)
        self.affine_weight = nn.Parameter(torch.ones(self.num_features))
        self.affine_bias = nn.Parameter(torch.zeros(self.num_features))

    def _get_statistics(self, x):
        dim2reduce = tuple(range(1, x.ndim-1))
        self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach() # [B, 1, C]
        self.stdev = torch.sqrt(torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps).detach() # [B, 1, C]

    def _normalize(self, x):
        # B,L,C
        x = x - self.mean 
        x = x / self.stdev
        if self.affine:
            x = x * self.affine_weight
            x = x + self.affine_bias

        return x

    def _denormalize(self, x):
        if self.affine:
            x = x - self.affine_bias
            x = x / (self.affine_weight + self.eps*self.eps)
        x = x * self.stdev
        x = x + self.mean
        
        return x
    
class sample_norm(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.norm_type = configs.norm_type
        self.seq_len = configs.seq_len
        self.channels = configs.enc_in

        if self.norm_type == 'sample_learnable':
            assert self.seq_len is not None, "sample_learnable need seq_len"
            self.stat_net = nn.Sequential(
                nn.Linear(self.seq_len, 2)
            )            

        self.center = None 
        self.scale = None  

    def normalize(self, x):
        # x: [B, L, C]
        if self.norm_type == 'median' or self.norm_type == 'Median':
            self.center = torch.median(x, dim=1, keepdim=True).values
            q1 = torch.quantile(x, 0.25, dim=1, keepdim=True)
            q3 = torch.quantile(x, 0.75, dim=1, keepdim=True)
            self.scale = q3 - q1 + 1e-7
            x = (x - self.center) / self.scale

        elif self.norm_type == 'standard':
            self.center = torch.mean(x, dim=1, keepdim=True)
            self.scale = torch.sqrt(torch.var(x, dim=1, keepdim=True) + 1e-7)
            x = (x - self.center) / self.scale

        elif self.norm_type == 'mean':
            self.center = torch.mean(x, dim=1, keepdim=True)
            self.scale = 1.0
            x = (x - self.center)

        elif self.norm_type == 'student-t':
            self.center = torch.median(x, dim=1, keepdim=True).values
            diff = torch.abs(x - self.center)
            self.scale = torch.median(diff, dim=1, keepdim=True).values * 1.4826 + 1e-7
            x = (x - self.center) / self.scale

        elif self.norm_type == 'sample_learnable':
            x_for_stat = x.permute(0, 2, 1) 
            mu, sigma = self.stat_net(x_for_stat).chunk(2, dim=-1)
            sigma = F.softplus(sigma) + 1e-7
            self.center = mu.permute(0, 2, 1)
            self.scale = sigma.permute(0, 2, 1)
            x = (x - self.center) / self.scale

        elif self.norm_type == 'none':
            pass
        return x

    def denormalize(self, y):
        if self.norm_type == 'none':
            return y
        if self.center is None or self.scale is None:
            raise RuntimeError("Must call normalize() before denormalize()")
        if self.norm_type == 'sample_learnable':
             y = y * torch.abs(self.scale) + self.center
        else:
             y = y * self.scale + self.center
        return y

    def forward(self, x, mode='norm'):
        if mode == 'norm':
            return self.normalize(x)
        elif mode == 'denorm':
            return self.denormalize(x)
        elif mode == 'none':
            return x
