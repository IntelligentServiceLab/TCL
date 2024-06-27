import os
import os.path as osp
from typing import Callable, List, Optional
import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import BertModel, BertTokenizer
import os
import glob

import pandas as pd
import torch

from torch_geometric.data import (
    HeteroData,
    InMemoryDataset,
    download_url,
    extract_zip,
)
from transformers import BertTokenizer, BertModel
from tqdm import tqdm
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

tokenizer = BertTokenizer.from_pretrained('./bert')
bert_model = BertModel.from_pretrained('./bert').cuda()


def check_for_pt_files(file_name,folder_path):
    if not os.path.exists('./Data/Train/'):
        os.makedirs('./Data/Train/')
    if not os.path.exists('./Data/Test/'):
        os.makedirs('./Data/Test/')
    # 使用 glob 模块匹配文件路径模式
    pt_files = glob.glob(os.path.join(folder_path, f'{file_name}.pt'))

    # 检查是否有匹配的文件
    if pt_files:
      return True
    else:
      return False

def DataLoad(type):
    data=pd.read_csv('programableWeb/finnnnnal2.csv', sep='\t', header=0)
    mashup=data.drop_duplicates(subset=["Mashup_id"]).sort_values("Mashup_id")
    api=data.drop_duplicates(subset=["API_id"]).sort_values("API_id")
    mashup_mapping = dict(zip(mashup['Mashup_id'], mashup['Mashup_name']))
    api_mapping=dict(zip(api["API_id"],api["API_name"]))
    src = data['Mashup_id'].tolist()
    dst = data['API_id'].tolist()

    edge_attr = torch.ones(len(src), dtype=torch.bool).view(-1,1).to(torch.long)
    # 构建边索引
    edge_index = [[], []]
    for i in range(len(edge_attr)):
        if edge_attr[i]:
            edge_index[0].append(src[i])
            edge_index[1].append(dst[i])
    edge_index = torch.tensor(edge_index)
    api_weighted=pd.read_csv("./programableWeb/api_info_weight.csv",sep='\t',header=0)
    mashup_weighted=pd.read_csv("./programableWeb/mashup_info_weight.csv",sep='\t',header=0)
    if type=="Train":

        if check_for_pt_files("mashup_emb_train", "./Data/Train/"):
            print("存在训练数据，正在加载")
            Train_mashup_emb = torch.load('./Data/Train/mashup_emb_train.pt', map_location='cuda')
            Train_api_emb = torch.load('./Data/Train/api_emb_train.pt', map_location='cuda')
            a=1
        else:
            print("未检测到训练数据，BERT文本语义提取中")
            Train_mashup_emb = bert_convert_emb(data["Mashup_des"])
            Train_api_emb = bert_convert_emb(data["API_des"])
            torch.save(Train_mashup_emb, './Data/Train/mashup_emb_train.pt')
            torch.save(Train_api_emb, './Data/Train/api_emb_train.pt')
        return mashup_mapping,api_mapping,edge_index,Train_mashup_emb,Train_api_emb,mashup_weighted,api_weighted
    else:
        if check_for_pt_files("mashup_emb_Test", "./Data/Test/"):
            print("存在测试数据，正在加载")
            Test_mashup_emb = torch.load('Data/Test/mashup_emb_Test.pt', map_location='cuda')
            Test_api_emb = torch.load('Data/Test/api_emb_Test.pt', map_location='cuda')
        else:
            print("未检测到测试数据，BERT文本语义提取中")
            Test_mashup_emb = bert_convert_emb(mashup["Mashup_des"])
            Test_api_emb = bert_convert_emb(api["API_des"])
            torch.save(Test_mashup_emb, 'Data/Test/mashup_emb_Test.pt')
            torch.save(Test_api_emb, 'Data/Test/api_emb_Test.pt')

        max_mashup_id = data['Mashup_id'].max()
        max_api_id = data['API_id'].max()
        call_relation_labels_mashup = [[0] * (max_api_id + 1) for _ in range(max_mashup_id + 1)]

        # 遍历数据集中的每一行，将调用关系标签矩阵中对应位置的值设为1
        for _, row in data.iterrows():
            mashup_id = row['Mashup_id']
            api_id = row['API_id']
            call_relation_labels_mashup[mashup_id][api_id] = 1
        return mashup_mapping, api_mapping, edge_index, Test_mashup_emb, Test_api_emb, call_relation_labels_mashup

def encode_data(descriptions):
    input_ids = []
    attention_masks = []
    for desc in descriptions:
        encoded_desc = tokenizer.encode_plus(desc, add_special_tokens=True, max_length=150,
                                             padding='max_length',
                                             truncation=True, return_attention_mask=True, return_tensors='pt')
        input_ids.append(encoded_desc['input_ids'])
        attention_masks.append(encoded_desc['attention_mask'])
    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)
    return input_ids.to(device), attention_masks.to(device)  # 将数据移动到 CUDA 设备上

def bert_convert_emb(descriptions):

    input_ids, attention_masks = encode_data(descriptions)

    # 将 input_ids 和 attention_masks 封装成 TensorDataset
    dataset = TensorDataset(input_ids, attention_masks)

    # 定义批次大小
    batch_size = 512

    # 创建 DataLoader 对象，用于分批次加载数据
    data_loader = DataLoader(dataset, batch_size=batch_size)

    # 获取总批次数量
    total_batches = len(data_loader)

    # 遍历每个批次，并在每个批次上进行模型推理
    all_sentence_vectors = []
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(data_loader, desc="Processing")):
            batch_input_ids, batch_attention_masks = batch
            # 将 input_ids 和 attention_masks 传入 BERT 模型
            outputs = bert_model(batch_input_ids, attention_mask=batch_attention_masks)
            # 获取 BERT 模型的文本向量表示（最后一层的隐藏状态）
            batch_sentence_vectors = outputs[1] # 获取 [CLS] token 的隐藏状态作为句子向量
            all_sentence_vectors.append(batch_sentence_vectors)
            # 更新进度条
            tqdm.write(f"Processed batch {batch_idx + 1}/{total_batches}")

    # 将每个批次的文本向量拼接起来
    all_sentence_vectors = torch.cat(all_sentence_vectors, dim=0)

    # all_sentence_vectors 就是每个描述文本的 BERT 表示（文本向量）

    return all_sentence_vectors
