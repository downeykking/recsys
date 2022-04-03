import torch
import torch.nn as nn
import torch.nn.functional as F
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


#
class InteractLayer(nn.Module):
    def __init__(self, embed_size, n_heads=2, use_res=True, scaling=False):
        super(InteractLayer, self).__init__()

        self.n_heads = n_heads
        self.att_embed_size = embed_size // n_heads
        self.use_res = use_res
        self.scaling = scaling

        self.W_Q = nn.Linear(embed_size, embed_size, bias=False)
        self.W_K = nn.Linear(embed_size, embed_size, bias=False)
        self.W_V = nn.Linear(embed_size, embed_size, bias=False)

        if self.use_res:
            self.W_Res = nn.Parameter(torch.Tensor(embed_size, embed_size))
        for tensor in self.parameters():
            nn.init.normal_(tensor, mean=0.0, std=0.05)
        self.activate = nn.ReLU()

    def forward(self, input_x):
        """attention layer interacting

        Args:
            input_x (): (bs, field, embed_size)

        Returns:
            _type_: [batch_size, field, n_heads * att_emb_size]
        """
        residual, batch_size = input_x, input_x.size(0)
        # (bs, field, embed_size) dot (embed_size, att_embed_size * n_heads)
        # (bs, field, att_embed_size * n_heads)
        Q = self.W_Q(input_x)
        K = self.W_K(input_x)
        V = self.W_V(input_x)

        # [batch size, n heads, field, att_embed_size]
        # why do this see https://github.com/bentrevett/pytorch-seq2seq/issues/148
        Q = Q.view(batch_size, -1, self.n_heads,
                   self.att_embed_size).permute(0, 2, 1, 3)
        K = K.view(batch_size, -1, self.n_heads,
                   self.att_embed_size).permute(0, 2, 1, 3)
        V = V.view(batch_size, -1, self.n_heads,
                   self.att_embed_size).permute(0, 2, 1, 3)

        # inner score
        # [batch size, n heads, field, att_embed_size] dot [batch size, n heads, att_embed_size, field]
        # [batch size, n heads, field, field]
        scores = torch.matmul(Q, K.transpose(-1, -2))
        if self.scaling:
            scores /= self.att_embed_size**0.5

        normalized_att_scores = F.softmax(scores, dim=-1)

        # [batch_size, n_heads, field, field] dot [batch_size, n_heads, field, att_emb_size]
        # [batch_size, n_heads, field, att_emb_size]
        multi_attention_output = torch.matmul(normalized_att_scores, V)

        # [batch_size, field, n_heads * att_emb_size]
        multi_attention_output = multi_attention_output.transpose(1, 2).reshape(
            batch_size, -1, self.att_embed_size * self.n_heads)

        # 加入残差 return [batch_size, field, n_heads * att_emb_size]
        if self.use_res:
            multi_attention_output += torch.matmul(residual, self.W_Res)
        output = self.activate(multi_attention_output)
        return output


class AutoInt(nn.Module):
    def __init__(self,
                 feat_size,
                 feature_columns,
                 emb_size=8,
                 hidden_dims=[256, 128],
                 att_layer_num=3,
                 att_head_num=2,
                 att_res=True,
                 l2_reg=0.00001,
                 init_std=0.0001,
                 dropout=0.5,
                 dnn_dropout=0.5):
        """AutoInt MODEL

        Args:
            feat_size (dict: keys:feature_name, values:feature_num): dict, 每个特征对应的具体类别数量
            feature_columns (list of tuples): 特征及每个特征属于dense还是sparse
            emb_size (int): 嵌入维度, 即论文中k, Defaults to 8
            att_layer_num (int): attention层数, Defaults to 3
            att_head_num (int): attention多头数, Defaults to 2
            att_res (bool): 是否使用残差, Defaults to True
            l2_reg (float, optional): l2正则化. Defaults to 0.00001.
            dropout (float, optional): dropout. Defaults to 0.9.
        """
        super(AutoInt, self).__init__()

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

        # AutoInt层
        self.int_layers = nn.ModuleList([
            InteractLayer(emb_size, att_head_num, att_res)
            for _ in range(att_layer_num)
        ])

        # dnn层
        self.dnn = DNN(len(self.dense_feature_columns) +
                       emb_size * len(self.sparse_feature_columns),
                       hidden_dims,
                       l2_reg=l2_reg,
                       dropout=dnn_dropout,
                       use_bn=False,
                       init_std=init_std)
        self.dnn_out_layer = nn.Linear(
            hidden_dims[-1] + emb_size * len(self.sparse_feature_columns),
            1,
            bias=False)

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

        sparse_input = torch.cat(sparse_embedding, dim=1)
        dense_input = torch.cat(dense_values, dim=1)

        # 一阶线性 (bs, 1)
        wide_input_x = X
        wide_logit = self.wide_linear(wide_input_x)
        wide_logit = self.act(wide_logit)
        wide_logit = self.dropout(wide_logit)

        # 生成AutoInt (bs, field, emb_size) -> (bs, field * emb_size)
        att_input = sparse_input
        for layer in self.int_layers:
            att_input = layer(att_input)
        # [batch_size, field * emb_size]
        att_output = torch.flatten(att_input, start_dim=1)

        # (bs, emb_size*field + dense_feature_num)
        sparse_input_flat = torch.flatten(sparse_input, start_dim=1)
        dnn_input = torch.cat([sparse_input_flat, dense_input], dim=1)
        dnn_output = self.dnn(dnn_input)

        # (bs, hidden[-1]+field * emb_size)
        stack_out = torch.cat([dnn_output, att_output], dim=1)
        autoint_logit = self.dnn_out_layer(stack_out)

        logit = self.sigmoid(wide_logit + autoint_logit)

        return logit.squeeze(1)
