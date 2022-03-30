import torch
import torch.nn as nn
from collections import defaultdict


class DNN(nn.Module):
    def __init__(self,
                 inputs_dim,
                 hidden_dims,
                 l2_reg=0,
                 dropout=0,
                 use_bn=False,
                 init_std=0.0001):
        super(DNN, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.l2_reg = l2_reg
        self.use_bn = use_bn
        hidden_dims = [inputs_dim] + hidden_dims
        self.linears = nn.ModuleList([
            nn.Linear(hidden_dims[i], hidden_dims[i + 1])
            for i in range(len(hidden_dims) - 1)
        ])
        if use_bn:
            self.bn = nn.ModuleList([
                nn.BatchNorm1d(hidden_dims[i], hidden_dims[i + 1])
                for i in range(len(hidden_dims) - 1)
            ])
        self.activation_layer = nn.ModuleList(
            [nn.ReLU() for _ in range(len(hidden_dims) - 1)])
        for name, tensor in self.linears.named_parameters():
            if 'weight' in name:
                nn.init.normal_(tensor, mean=0, std=init_std)

    def forward(self, inputs):
        # (bs, dense_dim + embed_size)
        deep_input = inputs
        for i in range(len(self.linears)):
            fc = self.linears[i](deep_input)
            if self.use_bn:
                fc = self.bn[i](fc)
            fc = self.activation_layer[i](fc)
            fc = self.dropout(fc)
            deep_input = fc
        return deep_input


# 相较于fm不同的地方在于生成的是 dim=k 的向量
class BiInteractionPooling(nn.Module):
    def __init__(self):
        super(BiInteractionPooling, self).__init__()

    def forward(self, inputs):
        # (bs, field, emb_size)
        fm_input = inputs
        square_of_sum = torch.pow(torch.sum(fm_input, dim=1), 2)
        sum_of_square = torch.sum(torch.pow(fm_input, 2), dim=1)
        cross_term = square_of_sum - sum_of_square
        cross_term = 0.5 * cross_term
        # return (bs, emb_size)
        return cross_term


class NFM(nn.Module):
    def __init__(self,
                 feat_size,
                 feature_columns,
                 emb_size=4,
                 hidden_dims=[256, 128],
                 l2_reg=0.00001,
                 init_std=0.0001,
                 dropout=0.5,
                 bi_dropout=0.9,
                 dnn_dropout=0.5):
        """NFM MODEL

        Args:
            feat_size (dict: keys:feature_name, values:feature_num): dict, 每个特征对应的具体类别数量
            feature_columns (list of tuples): 特征及每个特征属于dense还是sparse
            emb_size (int): 嵌入维度, 即论文中k, Defaults to 8
            l2_reg (float, optional): l2正则化. Defaults to 0.00001.
            dropout (float, optional): dropout. Defaults to 0.9.
        """
        super(NFM, self).__init__()

        self.sparse_feature_columns = list(
            filter(lambda x: x[1] == 'sparse', feature_columns))

        self.dense_feature_columns = list(
            filter(lambda x: x[1] == 'dense', feature_columns))

        self.embedding_dic = nn.ModuleDict({
            feat[0]: nn.Embedding(feat_size[feat[0]], emb_size, sparse=False)
            for feat in self.sparse_feature_columns
        })

        for tensor in self.embedding_dic.values():
            nn.init.normal_(tensor.weight, mean=0, std=init_std)

        # 用来判断取原始feature中的哪一列数据
        self.feature_index = defaultdict(int)
        start = 0
        for feat in feat_size:
            self.feature_index[feat] = start
            start += 1

        # 一阶线性层
        self.wide_linear = nn.Linear(len(feature_columns), 1)

        for name, tensor in self.wide_linear.named_parameters():
            if 'weight' in name:
                nn.init.normal_(tensor, mean=0, std=0.00001)

        # nfm层
        self.bi_pooling = BiInteractionPooling()
        if bi_dropout > 0:
            self.bi_dropout = nn.Dropout(bi_dropout)

        # nfm后面接dnn层
        self.dnn = DNN(len(self.dense_feature_columns) + emb_size,
                       hidden_dims,
                       l2_reg=l2_reg,
                       dropout=dnn_dropout,
                       use_bn=False,
                       init_std=init_std)
        self.dnn_out_layer = nn.Linear(hidden_dims[-1], 1, bias=False)

        self.sigmoid = nn.Sigmoid()
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, X):
        # sparse_embedding[0].size()  (bs, 1, emb_size)
        sparse_embedding = [
            self.embedding_dic[feat[0]](
                X[:, self.feature_index[feat[0]]].long()).unsqueeze(1)
            for feat in self.sparse_feature_columns
        ]

        dense_values = [
            X[:, self.feature_index[feat[0]]].unsqueeze(1)
            for feat in self.dense_feature_columns
        ]

        dense_input = torch.cat(dense_values, dim=1)

        # 一阶线性 (bs, 1)
        wide_input_x = X
        wide_logit = self.wide_linear(wide_input_x)
        wide_logit = self.act(wide_logit)
        wide_logit = self.dropout(wide_logit)

        # 生成nfm (bs, field, emb_size) -> (bs, emb_size)
        nfm_input = torch.cat(sparse_embedding, dim=1)
        bi_out = self.bi_pooling(nfm_input)

        if self.bi_dropout:
            bi_out = self.bi_dropout(bi_out)

        # bi_out = torch.flatten(torch.cat([bi_out], dim=-1), start_dim=1)

        # (bs, emb_size + dense_feature_num)
        dnn_input = torch.cat([dense_input, bi_out], dim=1)
        dnn_output = self.dnn(dnn_input)
        dnn_logit = self.dnn_out_layer(dnn_output)

        logit = self.sigmoid(wide_logit + dnn_logit)

        return logit.squeeze(1)
