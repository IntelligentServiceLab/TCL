import pandas as pd
import csv
from dataset import DataLoad
from torch_sparse import SparseTensor
from sklearn.model_selection import train_test_split
from model import Model_Multiple
from sanfm import SANFM, snafm_loss
from utils import bpr_loss, my_sample_mini_batch, ModelConfig, normalize_sample_label,\
    CL_loss222
import torch
from tqdm import tqdm
import numpy as np

from itertools import chain
import os

config = ModelConfig()


def train(sanfm_emb, lgc_emb, cl_rate, temp, dropout):
    print("开始训练")
    config = ModelConfig()
    myseed = 3030
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    mashup_mapping, api_mapping, edge_index, mashup_emb, api_emb, mashup_weighted, api_weighted = DataLoad("Train")
    mashup_mapping_test, api_mapping_test, edge_index_test, mashup_emb_test, api_emb_test, test_label_test = DataLoad(
        "Test")

    with open('label.csv', 'w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerows(test_label_test)



    num_mashup = len(mashup_mapping)
    num_api = len(api_mapping)
    num_edges = len(edge_index[1])
    all_index = [i for i in range(num_edges)]

    train_index, test_index = train_test_split(all_index, test_size=0.2, random_state=myseed)
    train_edge_index = edge_index[:, train_index]
    test_edge_index = edge_index[:, test_index]

    train_sparse_edge_index = SparseTensor(row=train_edge_index[0], col=train_edge_index[1],
                                           sparse_sizes=(num_api + num_mashup, num_api + num_mashup)).to(device)
    test_sparse_edge_index = SparseTensor(row=test_edge_index[0], col=test_edge_index[1],
                                          sparse_sizes=(num_api + num_mashup, num_api + num_mashup)).to(device)

    model = Model_Multiple(num_mashup, num_api, embedding_dim=lgc_emb, num_layers=config.num_layers,
                           add_self_loops=False, input_dim=config.input_dim, hidden_dim=lgc_emb)
    sanfm_obj = SANFM(embed_dim=sanfm_emb, droprate=dropout, i_num=lgc_emb * 4)

    model = model.to(device)
    sanfm_obj = sanfm_obj.to(device)
    optimizer = torch.optim.Adam(params=chain(model.parameters(), sanfm_obj.parameters()), lr=config.lr,
                                 weight_decay=config.weight_decay)

    # 初始化最佳综合指标
    best_combined_score = 0  # 初始值为负无穷大
    Hit = 0
    Recall = 0
    NDCG = 0

    for epoch in tqdm(range(1, config.n_epoch + 1)):
        model.train()
        sanfm_obj.train()
        optimizer.zero_grad()
        user_indices, pos_item_indices, neg_item_indices, mashup_e, api_e, neg_api_e = my_sample_mini_batch(
            config.train_batch_size, train_edge_index, mashup_emb, api_emb)

        users_emb_final, users_emb_0, items_emb_final, items_emb_0, api_pooled_output, mashup_pooled_output, neg_api_pooled_output = model(
            train_sparse_edge_index, mashup_e, api_e, neg_api_e)
        # batch_size
        users_emb_final = users_emb_final[user_indices]
        users_emb_0 = users_emb_0[user_indices]
        pos_items_emb_final = items_emb_final[pos_item_indices]
        pos_items_emb_0 = items_emb_0[pos_item_indices]
        neg_items_emb_final = items_emb_final[neg_item_indices]
        neg_items_emb_0 = items_emb_0[neg_item_indices]
        train_loss = bpr_loss(users_emb_final, users_emb_0, pos_items_emb_final, pos_items_emb_0, neg_items_emb_final,
                              neg_items_emb_0, config.lamda)

        # batch_size_contrastive_learning
        # 拿api和mashup的index转换为node_id，然后获取其权重
        mashup_node_list = train_edge_index[0, user_indices].tolist()
        api_node_list = train_edge_index[1, pos_item_indices].tolist()

        mashup_weighted_selected = pd.DataFrame({'Mashup_id': mashup_node_list})
        api_weighted_selected = pd.DataFrame({'API_id': api_node_list})

        mashup_weighted_loss = mashup_weighted_selected.merge(mashup_weighted, on='Mashup_id', how='left')[
            'loss_weight'].tolist()
        api_weighted_loss = api_weighted_selected.merge(api_weighted, on='API_id', how='left')['loss_weight'].tolist()

        # cl_loss_m=CL_loss(mashup_pooled_output,users_emb_final,temperature=temp)
        # cl_loss_a=CL_loss(api_pooled_output,pos_items_emb_final,temperature=temp)

        mashup_weighted_loss = torch.tensor(mashup_weighted_loss).to(device)
        api_weighted_loss = torch.tensor(api_weighted_loss).to(device)

        cl_loss_m = CL_loss222(mashup_pooled_output, users_emb_final, temperature=temp, weight=mashup_weighted_loss,
                               b_cos=True)
        cl_loss_a = CL_loss222(api_pooled_output, pos_items_emb_final, temperature=temp, weight=api_weighted_loss,
                               b_cos=True)

        mashup_emb_temp = torch.cat((mashup_pooled_output, users_emb_final), dim=1)
        api_emb_temp = torch.cat((api_pooled_output, pos_items_emb_final), dim=1)
        neg_api_emb_temp = torch.cat((neg_api_pooled_output, neg_items_emb_final), dim=1)

        sample, label = normalize_sample_label(mashup_emb_temp, api_emb_temp, neg_api_emb_temp)

        rec_loss = snafm_loss(sanfm_obj, sample, label)

        Totle_loss = train_loss + rec_loss + cl_rate * (cl_loss_m + cl_loss_a)

        Totle_loss.backward(retain_graph=True)
        optimizer.step()
        if (epoch % config.eval_steps == 0):
            model.eval()
            sanfm_obj.eval()
            Hit_rate, precision_average, recall_average, f1_score_average, MAP_average, ndcg_average, loss, cl_loss_m, cl_loss_a, rec_loss = my_evaluation_4(
                model, edge_index, train_edge_index, test_edge_index, test_sparse_edge_index, config.K, config.lamda,
                mashup_emb_test, api_emb_test, temp, sanfm_obj, config.test_batch_size, test_label_test)

            print(
                f"HR={Hit_rate:.5f},precision={precision_average:.5f}, recall={recall_average:.5f}, f1_score={f1_score_average:.5f}, MAP={MAP_average:.5f}, ndcg={ndcg_average:.5f},rec_loss={rec_loss:.5f},loss={loss:.5f}")

            # 计算综合指标
            combined_score =Hit_rate+recall_average+ndcg_average

            # 检查是否为最佳结果
            if combined_score >= best_combined_score:
                # print(f"combined_score={combined_score},best_combined_score={best_combined_score}")
                best_combined_score = combined_score
                Hit = Hit_rate
                Recall = recall_average
                NDCG = ndcg_average
    print(f"HR={Hit:.5f},Recall={Recall:.5f}, NDCG={NDCG:.5f}")
    return Hit, Recall, NDCG

train(32,16,0.01,0.01,0.5)
