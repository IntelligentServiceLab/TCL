
import torch
import torch.nn.functional as F

import torch.nn as nn
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class SANFM(nn.Module):
    def __init__(self, embed_dim, droprate=0.5, i_num=None):
        super(SANFM, self).__init__()

        # 初始化模型参数
        self.i_num = i_num  # 输入特征数量 1536
        self.embed_dim = embed_dim  # 设定嵌入维度为64
        self.att_dim = embed_dim  # 用于 selfatt 的输出维度，可修改
        self.bi_inter_dim = embed_dim  # 用于pairwise interaction的维度 可修改
        self.droprate = droprate  # 剪枝率
        self.criterion = nn.BCELoss(weight=None, reduction='mean')  # 二分类交叉熵损失函数
        self.sigmoid = nn.Sigmoid()

        # 定义模型的各个层
        self.dense_embed = nn.Linear(self.i_num, self.embed_dim)  # 将输入转化为指定维度的嵌入向量

        self.pairwise_inter_v = nn.Parameter(torch.empty(self.embed_dim, self.bi_inter_dim))  # 用于pairwise interaction

        # Self-Attention所需参数
        self.query_matrix = nn.Parameter(torch.empty(self.embed_dim, self.att_dim))
        self.key_matrix = nn.Parameter(torch.empty(self.embed_dim, self.att_dim))
        self.value_matrix = nn.Parameter(torch.empty(self.embed_dim, self.att_dim))
        self.softmax = nn.Softmax(dim=-1)  # Softmax函数

        # MLP层
        self.hidden_1 = nn.Linear(self.embed_dim, self.embed_dim, bias=True)
        self.hidden_2 = nn.Linear(self.embed_dim, 1)

        self.bn = nn.BatchNorm1d(self.embed_dim, momentum=0.9)  # Batch normalization

        self._init_weight_()  # 初始化模型参数

    def BiInteractionPooling(self, pairwise_inter):
        # 计算二阶交互特征
        inter_part1_sum = torch.sum(pairwise_inter, dim=1)
        inter_part1_sum_square = torch.square(inter_part1_sum)  # square_of_sum

        inter_part2 = pairwise_inter * pairwise_inter
        inter_part2_sum = torch.sum(inter_part2, dim=1)  # sum of square
        bi_inter_out = 0.5 * (inter_part1_sum_square - inter_part2_sum)
        return bi_inter_out

    def _init_weight_(self):
        """初始化模型参数"""
        # dense embedding
        nn.init.normal_(self.dense_embed.weight, std=0.1)
        # pairwise interaction pooling
        nn.init.normal_(self.pairwise_inter_v, std=0.1)
        # deep layers
        nn.init.kaiming_normal_(self.hidden_1.weight)
        nn.init.kaiming_normal_(self.hidden_2.weight)
        # attention part
        nn.init.kaiming_normal_(self.query_matrix)
        nn.init.kaiming_normal_(self.key_matrix)
        nn.init.kaiming_normal_(self.value_matrix)

    def forward(self, batch_data):  # 前向传播
        batch_data = batch_data.to(torch.float32)

        # Embedding部分
        dense_embed = self.dense_embed(batch_data)

        # Interaction部分
        pairwise_inter = dense_embed.unsqueeze(1) * self.pairwise_inter_v
        pooling_out = self.BiInteractionPooling(pairwise_inter)

        # Self-Attention部分
        X = pooling_out
        proj_query = torch.mm(X, self.query_matrix)
        proj_key = torch.mm(X, self.key_matrix)
        proj_value = torch.mm(X, self.value_matrix)

        S = torch.mm(proj_query, proj_key.T)
        attention_map = self.softmax(S)

        value_weight = proj_value[:, None] * attention_map.T[:, :, None]
        value_weight_sum = value_weight.sum(dim=0)

        # MLP部分
        mlp_hidden_1 = F.relu(self.bn(self.hidden_1(value_weight_sum)))
        mlp_hidden_2 = F.dropout(mlp_hidden_1, training=self.training, p=self.droprate)
        mlp_out = self.hidden_2(mlp_hidden_2)
        final_sig_out = self.sigmoid(mlp_out)
        final_sig_out_squeeze = final_sig_out.squeeze()
        return final_sig_out_squeeze

    def loss(self, batch_input, batch_label):
        pred = self.forward(batch_input)
        pred = pred.to(torch.float32).to(device)
        batch_label = batch_label.to(torch.float32).squeeze().to(device)
        loss1 = self.criterion(pred, batch_label)  # 计算损失
        return loss1
    def test_loss(self, pred, batch_label):

        pred = pred.to(torch.float32).to(device)
        batch_label = batch_label.to(torch.float32).squeeze().to(device)
        loss1 = self.criterion(pred, batch_label)  # 计算损失
        return loss1
    # pred = out.to(torch.float32).to(device)
    # batch_label = label.to(torch.float32).squeeze().to(device)

def snafm_loss(model, sample, label):
    loss2 = model.loss(sample, label)
    return loss2
