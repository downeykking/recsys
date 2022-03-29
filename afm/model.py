import torch
import torch.nn as nn
import itertools
from collections import defaultdict


class FM(nn.Module):
    def __init__(self):
        super(FM, self).__init__()

    def forward(self, inputs):
        # (bs, field, emb_size)
        fm_input = inputs
        square_of_sum = torch.pow(torch.sum(fm_input, dim=1), 2)
        sum_of_square = torch.sum(torch.pow(fm_input, 2), dim=1)
        cross_term = square_of_sum - sum_of_square
        cross_term = 0.5 * torch.sum(cross_term, dim=1, keepdim=True)
        return cross_term


class AFMLayer(nn.Module):
    def __init__(self, emb_size, attention_factor=4, l2_reg=0.0, dropout=0.0):
        super(AFMLayer, self).__init__()

        self.l2_reg = l2_reg

        self.attention_W = nn.Parameter(torch.Tensor(emb_size,
                                                     attention_factor))
        self.attention_b = nn.Parameter(torch.Tensor(attention_factor))
        self.projection_h = nn.Parameter(torch.Tensor(attention_factor, 1))
        self.projection_p = nn.Parameter(torch.Tensor(emb_size, 1))

        for tensor in [self.attention_W, self.projection_h, self.projection_p]:
            nn.init.xavier_normal_(tensor, )
        for tensor in [self.attention_b]:
            nn.init.zeros_(tensor, )

        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, inputs):
        # inputs is list of (bs, 1, emb_size)
        embed_vec_list = inputs
        row, col = [], []

        for r, c in itertools.combinations(embed_vec_list, 2):
            row.append(r)
            col.append(c)

        p = torch.cat(row, dim=1)
        q = torch.cat(col, dim=1)
        inner_product = p * q

        # (v_i \odot v_j)x_ix_j
        # (bs, f*(f-1)/2, emb_size)
        bi_interaction = inner_product

        # \{alpha_{ij}}^' = h^T*Relu(W(v_i \odot v_j)x_ix_j + b)
        # (bs, f*(f-1)/2, emb_size) (emb_size, atten_factor)
        attention_temp = self.relu(
            torch.matmul(bi_interaction, self.attention_W) + self.attention_b)

        #  \alpha_{ij} = softmax(\{alpha_{ij}}^')
        # (bs, f*(f-1)/2, atten_factor) (atten_factor, 1)
        normalized_att_score = self.softmax(
            torch.matmul(attention_temp, self.projection_h))

        # y' = \a_{ij} (v_i \odot v_j)x_ix_j
        # (bs, f*(f-1)/2, 1) * (bs, f*(f-1)/2, emb_size) -> (bs, f*(f-1)/2, emb_size)
        # sum(bs, f*(f-1)/2, emb_size) -> (bs, emb_size)
        attention_output = torch.sum(normalized_att_score * bi_interaction,
                                     dim=1)

        attention_output = self.dropout(attention_output)

        # (bs, emb_size) dot (emb_size, 1)  -> (bs, 1)
        afm_out = torch.matmul(attention_output, self.projection_p)

        return afm_out


class AFM(nn.Module):
    def __init__(self,
                 feat_size,
                 feature_columns,
                 emb_size=4,
                 use_attention=True,
                 attention_factor=8,
                 l2_reg=0.00001,
                 dropout=0.9):
        """AFM MODEL

        Args:
            feat_size (dict: keys:feature_name, values:feature_num): dict, 每个特征对应的具体类别数量
            feature_columns (list of tuples): 特征及每个特征属于dense还是sparse
            emb_size (int): 嵌入维度, 即论文中k, Defaults to 8
            use_attention (bool, optional): 是否使用注意力机制. Defaults to True.
            attention_factor (int, optional): 注意力机制映射的size. Defaults to 8.
            l2_reg (float, optional): l2正则化. Defaults to 0.00001.
            dropout (float, optional): dropout. Defaults to 0.9.
        """
        super(AFM, self).__init__()

        self.sparse_feature_columns = list(
            filter(lambda x: x[1] == 'sparse', feature_columns))

        self.embedding_dic = nn.ModuleDict({
            feat[0]: nn.Embedding(feat_size[feat[0]], emb_size, sparse=False)
            for feat in self.sparse_feature_columns
        })

        # 用来判断取原始feature中的哪一列数据
        self.feature_index = defaultdict(int)
        start = 0
        for feat in feat_size:
            self.feature_index[feat] = start
            start += 1

        self.use_attention = use_attention
        if self.use_attention:
            self.fm = AFMLayer(emb_size, attention_factor, l2_reg, dropout)
        else:
            self.fm = FM()

        self.wide_linear = nn.Linear(len(feature_columns), 1)

        for name, tensor in self.wide_linear.named_parameters():
            if 'weight' in name:
                nn.init.normal_(tensor, mean=0, std=0.00001)

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

        wide_input_x = X
        wide_logit = self.wide_linear(wide_input_x)
        wide_logit = self.act(wide_logit)
        wide_logit = self.dropout(wide_logit)

        if self.use_attention:
            fm_logit = self.fm(sparse_embedding)
        else:
            fm_logit = self.fm(torch.cat(sparse_embedding, dim=1))

        logit = self.sigmoid(wide_logit + fm_logit)

        return logit.squeeze(1)
