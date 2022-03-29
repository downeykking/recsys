import torch
from torch import nn


class DeepFM(nn.Module):
    def __init__(self,
                 sparse_feature_columns,
                 dense_feature_size=0,
                 emb_size=8,
                 device=None,
                 hidden_dims=[256, 128],
                 num_classes=1,
                 dropout=0.2,
                 init_std=0.0001):
        """ DeepFM

        Args:
            sparse_feature_columns (list of int): 离散特征每个特征有多少类别
            dense_feature_size (int): 连续特征数量
            emb_size (int): 嵌入维度, 即论文中k, Defaults to 8
            hidden_dims (list, optional): dnn隐藏单元. Defaults to [256, 128].
            num_classes (int, optional): 输出层. Defaults to 1.
            dropout (int, optional): dropout. Defaults to 0.2.
        """

        super(DeepFM, self).__init__()
        self.sparse_feature_columns = sparse_feature_columns
        self.dense_feature_size = dense_feature_size
        self.emb_size = emb_size
        self.device = device
        self.hidden_dims = hidden_dims
        self.num_classes = num_classes
        self.dropout = nn.Dropout(dropout)
        self.init_std = init_std

        # dense特征一阶表示
        if self.dense_feature_size != 0:
            self.linear_dense = nn.Linear(self.dense_feature_size, 1)

        # sparse特征一阶表示
        self.linear_sparse_emb = nn.ModuleList(
            [nn.Embedding(voc_size, 1) for voc_size in sparse_feature_columns])

        # sparse特征二阶表示
        self.fm_2nd_order_sparse_emb = nn.ModuleList([
            nn.Embedding(voc_size, emb_size)
            for voc_size in sparse_feature_columns
        ])

        # dnn部分
        # dnn_input_size （26*8 + 13）
        self.dnn_input_size = self.emb_size * len(
            self.sparse_feature_columns) + self.dense_feature_size

        self.hidden_dims = [self.dnn_input_size] + self.hidden_dims

        self.linears = nn.ModuleList([
            nn.Linear(self.hidden_dims[i], self.hidden_dims[i + 1])
            for i in range(len(self.hidden_dims) - 1)
        ])

        self.relus = nn.ModuleList(
            [nn.ReLU() for _ in range(len(self.hidden_dims) - 1)])

        for name, tensor in self.linears.named_parameters():
            if 'weight' in name:
                nn.init.normal_(tensor, mean=0, std=self.init_std)

        self.dnn_out_layer = nn.Linear(self.hidden_dims[-1],
                                       self.num_classes,
                                       bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, X_sparse, X_dense=None):
        """
        X_sparse: sparse_feature [batch_size, sparse_feature_num]
        X_dense: dense_feature  [batch_size, dense_feature_num]
        """
        """FM部分"""
        # 一阶  包含sparse_feature和dense_feature的一阶
        # sparse特征嵌入成一维度 torch.Size([bs, 1, 1])
        linear_sparse = [
            emb(X_sparse[:, i]).unsqueeze(1)
            for i, emb in enumerate(self.linear_sparse_emb)
        ]

        # torch.Size([bs, 26, 1])
        linear_sparse = torch.cat(linear_sparse, dim=1)

        # [bs, 1, 1] 将sparse_feature sum后变成[bs, 1, 1]再降维[bs, 1]
        linear_sparse = torch.sum(linear_sparse, dim=1,
                                  keepdim=True).squeeze(-1)

        if X_dense is not None:
            linear_dense = self.linear_dense(X_dense)  # 将dense_feature压到一维度
            linear_part = linear_sparse + linear_dense
        else:
            linear_part = linear_sparse

        # 二阶sparse embed后生成 torch.Size([bs, 8]) 中间扩充一维变成 torch.Size([bs, 1, 8]) 方便后面concat
        fm_2nd_order_sparse = [
            emb(X_sparse[:, i]).unsqueeze(1)
            for i, emb in enumerate(self.fm_2nd_order_sparse_emb)
        ]

        # batch_size, sparse_feature_nums, emb_size  torch.Size([bs, 26, 8])
        fm_2nd_order_sparse = torch.cat(fm_2nd_order_sparse, dim=1)

        # 先求和再平方 batch_size, emb_size torch.Size([bs, 8])
        square_of_sum = torch.pow(torch.sum(fm_2nd_order_sparse, dim=1), 2)

        # 先平方再求和 batch_size, emb_size torch.Size([bs, 8])
        sum_of_square = torch.sum(torch.pow(fm_2nd_order_sparse, 2), dim=1)

        # 再求和
        cross_term = torch.sub(square_of_sum, sum_of_square)
        cross_term = 0.5 * torch.sum(cross_term, dim=1, keepdim=True)

        # batch_size, 1
        fm_logit = linear_part + cross_term

        # (bs, 26, embed) -> (bs, 26*embed)
        dnn_sparse = torch.flatten(fm_2nd_order_sparse, start_dim=1, end_dim=2)
        # print(dnn_sparse.size())
        # print(X_dense.size())

        # (bs, 13) concat (bs, 26*embed) -> (bs, 13+26*embed)
        dnn_input_x = torch.cat([dnn_sparse, X_dense], dim=1)

        for i in range(len(self.linears)):
            dnn_input_x = self.linears[i](dnn_input_x)
            dnn_input_x = self.relus[i](dnn_input_x)
            dnn_input_x = self.dropout(dnn_input_x)

        dnn_logit = self.dnn_out_layer(dnn_input_x)

        out = fm_logit + dnn_logit

        return self.sigmoid(out).squeeze(1)
