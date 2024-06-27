from torch import Tensor
from torch_sparse import SparseTensor, matmul
from torch_geometric.nn.conv import LGConv
from torch_geometric.typing import Adj

from torch_geometric.nn.conv import MessagePassing

import torch.nn as nn

import torch


class Model_Multiple(MessagePassing):
    def __init__(self, num_users, num_items, embedding_dim=64, num_layers=3, add_self_loops=False,input_dim=None,hidden_dim=None):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers
        self.add_self_loops = add_self_loops
        self.alpha = torch.tensor([1/(num_layers+1)]*(num_layers+1))
        self.users_emb = nn.Embedding(num_embeddings=self.num_users, embedding_dim=self.embedding_dim) # e_u^0
        self.items_emb = nn.Embedding(num_embeddings=self.num_items, embedding_dim=self.embedding_dim) # e_i^0
        nn.init.normal_(self.users_emb.weight, std=0.1)
        nn.init.normal_(self.items_emb.weight, std=0.1)
        self.convs = nn.ModuleList([LGConv() for _ in range(num_layers)])
        self.reset_parameters()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.mashup_fc = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.Tanh()
        self.tanh = nn.Tanh()
        self.dropout = nn.Dropout(0.5)
        self.api_fc=nn.Linear(input_dim,hidden_dim)
        self.api_layers = nn.Sequential(
            self.api_fc,
            self.relu,
            self.dropout
        )
        self.mashup_layers = nn.Sequential(

            self.mashup_fc,
            self.relu,
            self.dropout
        )

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.users_emb.weight)
        torch.nn.init.xavier_uniform_(self.items_emb.weight)
        for conv in self.convs:
            conv.reset_parameters()
    def forward(self, edge_index: SparseTensor, mashup_emb=None, api_emb=None, neg_api_emb=None):
        edge_index_norm = edge_index
        out = self.get_embedding(edge_index_norm)
        users_emb_final, items_emb_final = torch.split(out, [self.num_users, self.num_items])
        api_pooled_output = self.process_input(api_emb, "api")
        mashup_pooled_output = self.process_input(mashup_emb, "mashup")
        if neg_api_emb !=None :
            neg_api_pooled_output=self.process_input(neg_api_emb,"api")
            return users_emb_final, self.users_emb.weight, items_emb_final, self.items_emb.weight,api_pooled_output, mashup_pooled_output,neg_api_pooled_output
        else:
            return users_emb_final, self.users_emb.weight, items_emb_final, self.items_emb.weight, api_pooled_output, mashup_pooled_output

    def process_input(self, emb,type=None):

        if type=="api":
            processed_output = self.api_layers(emb)
        elif type=="mashup":
            processed_output=self.mashup_layers(emb)
        return processed_output


    def get_embedding(self, edge_index: Adj) -> Tensor:
        x = torch.cat([self.users_emb.weight, self.items_emb.weight])
        out = x * self.alpha[0]
        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index)
            out = out + x * self.alpha[i + 1]
        return out
