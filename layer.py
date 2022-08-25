import torch
import torch.nn as nn
import torch.sparse
from torch.nn.modules.module import Module


class IncidenceConvolution(Module):
    """Core operation of CNFNet"""

    def __init__(self, opt):
        super(IncidenceConvolution, self).__init__()
        self.opt = opt

        self.in_features = 1
        self.out_features = opt.num_feat
        self.fc_dim = opt.hidden

        self.func1 = nn.Linear(1, self.fc_dim)
        self.func2 = nn.Linear(self.fc_dim, self.fc_dim)
        self.func3 = nn.Linear(self.fc_dim, self.fc_dim)
        self.func4 = nn.Linear(self.fc_dim, 1)

    def forward(self, inc_m):
        """iterate instances and call energy function"""
        out = []
        for instance in inc_m: # itearate on each matrix data
            feat = []
            for arr in instance:
                feat.append(self._to_kernel(arr))
            out.append(torch.FloatTensor(feat))

        return torch.stack(out)

    def _fc_kernal(self, sig):
        """calculate the prediction given energy"""

        return self.func4(self.func3(self.func2(self.func1(sig))))

    def _to_kernel(self, arr):
        """calculate the energy representation of distribution"""
        # arr[0] is a vector
        l = torch.FloatTensor(arr[0])

        # normalization
        norm = l / torch.sum(l)

        # FC and product
        weight = torch.tensor([self._fc_kernal(_.unsqueeze(0)) for _ in norm])
        prod = torch.mul(weight, norm)

        # sum
        return torch.sum(prod)

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class FC(Module):
    """Fully connected layer for predicting runtime"""

    def __init__(self, opt):
        super(FC, self).__init__()
        self.opt = opt

        self.in_features = opt.energy_input_dim
        self.out_features = 2
        self.fc_dim = opt.hidden

        self.func1 = nn.Linear(self.in_features, self.fc_dim)
        self.func2 = nn.Linear(self.fc_dim, self.fc_dim)
        self.func3 = nn.Linear(self.fc_dim, self.fc_dim)
        self.func4 = nn.Linear(self.fc_dim, self.out_features)

    def forward(self, val):
        return  self.func4(self.func3(self.func2(self.func1(val))))

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'
