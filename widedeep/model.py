import torch
import torch.nn as nn


class WideDeep(nn.Module):
    def __init__(self,
                 sparse_feature_columns,
                 dense_feature_size=0,
                 emb_size=8,
                 hidden_dims=[256, 128],
                 num_classes=1,
                 dropout=0.2,
                 init_std=0.0001):
        """ WideDeep

        Args:
            sparse_feature_columns (list of int): 离散特征每个特征有多少类别
            dense_feature_size (int): 连续特征数量
            emb_size (int): 嵌入维度, 即论文中k, Defaults to 8
            hidden_dims (list, optional): dnn隐藏单元. Defaults to [256, 128].
            num_classes (int, optional): 输出层. Defaults to 1.
            dropout (int, optional): dropout. Defaults to 0.2.
        """

        super(WideDeep, self).__init__()
        self.sparse_feature_columns = sparse_feature_columns
        self.dense_feature_size = dense_feature_size

        # sparse特征嵌入
        self.sparse_emb = nn.ModuleList([
            nn.Embedding(voc_size, emb_size)
            for voc_size in sparse_feature_columns
        ])

        # Wide 不激活的情况下只输入连续型特征效果要好于一起输入sparse特征 激活后可以一起输入
        self.wide_linear = nn.Linear(len(sparse_feature_columns)+dense_feature_size, num_classes)

        # Deep
        # dnn_input_size （26*8 + 13）
        self.dnn_input_size = emb_size * len(
            sparse_feature_columns) + dense_feature_size

        self.hidden_dims = [self.dnn_input_size] + hidden_dims

        self.linears = nn.ModuleList([
            nn.Linear(self.hidden_dims[i], self.hidden_dims[i + 1])
            for i in range(len(self.hidden_dims) - 1)
        ])

        self.relus = nn.ModuleList(
            [nn.ReLU() for _ in range(len(self.hidden_dims) - 1)])

        self.dropout = nn.Dropout(dropout)

        for name, tensor in self.linears.named_parameters():
            if 'weight' in name:
                nn.init.normal_(tensor, mean=0, std=init_std)

        self.dnn_out_layer = nn.Linear(self.hidden_dims[-1],
                                       num_classes,
                                       bias=False)
        self.act = nn.ReLU()
        
        self.sigmoid = nn.Sigmoid()

    def forward(self, X_sparse, X_dense=None):
        """
        X_sparse: sparse_feature [batch_size, sparse_feature_num]
        X_dense: dense_feature  [batch_size, dense_feature_num]
        """

        # sparse embed后生成 torch.Size([bs, 8])
        sparse_embed = [
            emb(X_sparse[:, i]) for i, emb in enumerate(self.sparse_emb)
        ]

        # batch_size, sparse_feature_nums*emb_size  torch.Size([bs, 26*8])
        sparse_embed = torch.cat(sparse_embed, dim=1)

        # wide 激活的情况下只输入连续型特征效果要好于一起输入sparse特征 激活后可以一起输入
        wide_input_x = torch.cat([X_sparse, X_dense], dim=1)
        wide_logit = self.wide_linear(wide_input_x)
        wide_logit = self.act(wide_logit)
        wide_logit = self.dropout(wide_logit)

        # deep
        # (bs, 13) concat (bs, 26*embed) -> (bs, 13+26*embed)
        dnn_input_x = torch.cat([sparse_embed, X_dense], dim=1)

        for i in range(len(self.linears)):
            dnn_input_x = self.linears[i](dnn_input_x)
            dnn_input_x = self.relus[i](dnn_input_x)
            dnn_input_x = self.dropout(dnn_input_x)

        dnn_logit = self.dnn_out_layer(dnn_input_x)

        out = wide_logit + dnn_logit
        # out = dnn_logit
        return self.sigmoid(out).squeeze(1)
