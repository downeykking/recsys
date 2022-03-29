import torch
import torch.nn as nn


class FM(nn.module):
    def __init__(self, p, k):
        """
        Args:
            p (int): input_dim
            k (int): embed_dim
        """

        super(FM, self).__init__()
        self.p = p
        self.k = k
        self.linear = nn.Linear(self.p, 1, bias=True)
        self.v = nn.Parameter(torch.tensor(self.p, self.k), requires_grad=True)
        self.v.data.uniform_(-0.01, 0.01)
        self.drop = nn.Dropout(0.3)

    def forward(self, x):
        """fm function

        Args:
            x ([batch_size, dim]):  batch of features

        Returns:
            output (batch, 1): return value
        """
        linear_part = self.linear(x)
        inter_part1 = torch.pow(torch.mm(x, self.v), 2)
        inter_part2 = torch.mm(torch.pow(x, 2), torch.pow(self.v, 2))
        pair_interactions = 0.5 * torch.sum(torch.sub(inter_part1 - inter_part2), dim=1, keepdim=True)
        self.drop(pair_interactions)
        output = linear_part + pair_interactions
        return output
