import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math, copy, time
from utils import clones, attention

class MultiHeadAttention(nn.Module):
    def __init__(self, head, d_embedding, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        assert d_embedding % head ==0
        self.d_k = d_embedding // head
        self.head = head
        self.linears = clones(nn.Linear(d_embedding, d_embedding), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        # query: (batch, num_query, d_embedding)
        # key: (batch, num_key, d_embedding)
        # value: (batch, num_value, d_embedding)
        if mask is not None:
            mask = mask.unsqueeze(1)

        batch_size = query.size(0)

        # linear projection for query, key and value from (batch, num_word, d_embedding)
        #                                               to (batch, num_head, num_word, d_k)
        query = self.linears[0](query).view(batch_size, -1, self.head, self.d_k).transpose(1,2)
        key = self.linears[1](key).view(batch_size, -1, self.head, self.d_k).transpose(1,2)
        value = self.linears[2](value).view(batch_size, -1, self.head, self.d_k).transpose(1,2)
        print('query shape in multihead: {}'.format(query.shape))
        # Scaled Dot-Product Attention for each batch (batch, heads, num_word, d_k)
        x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)

        # "Concatenate" heads and apply final linear (batch, heads, num_word, d_k)
        #                                            =>(batch, num_word, d_embedding)
        x = x.transpose(1,2).contiguous().view(batch_size, -1, self.head*self.d_k)
        return self.linears[-1](x)

if __name__=='__main__':
    EMBEDDING_DIM = 512  # Embedding Size
    DIM_FF = 2048 # FeedForward dimension
    DIM_Q = DIM_K = DIM_V = 64  # dimension of K(=Q), V
    NUM_LAYER = 6  # number of Encoder and Decoder Layer
    NUM_HEAD = 8  # number of heads in Multi-Head Attention
    EXPECTED_MAX_NUM_WORD = 100 # ???
    BATCH_SIZE = 32

    torch.manual_seed(1)
    input = torch.randn(BATCH_SIZE, EXPECTED_MAX_NUM_WORD, EMBEDDING_DIM)
    w_qi = torch.randn(EMBEDDING_DIM, DIM_Q * NUM_HEAD)
    w_ki = torch.randn(EMBEDDING_DIM, DIM_K * NUM_HEAD)
    w_vi = torch.randn(EMBEDDING_DIM, DIM_V * NUM_HEAD)

    query = torch.matmul(input, w_qi)
    key = torch.matmul(input, w_ki)
    value = torch.matmul(input, w_vi)
    print("query shape before attention: {}".format(query.shape))

    mulhead = MultiHeadAttention(head=NUM_HEAD, d_embedding=EMBEDDING_DIM)
    score = mulhead(query, key, value)
    print('multi-head score size: {}'.format(score.shape))