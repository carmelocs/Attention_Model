# this file contains all the functions for other classes
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math, copy, time

def clones(module, N):
    "N identical layers"
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

def attention(query, key, value, mask=None, dropout=None):
    "Compute Scaled Dot-Product Attention"
    # query: (batch, num_head, num_query, d_q)
    # key: (batch, num_head, num_key, d_k)
    # value: (batch, num_head, num_value, d_v)
    # in which key and value are in pairs
    d_k = query.size(-1)
    # compute the similiarity (scores) of query and key by dot product
    # (batch, num_head, num_query, num_key)
    scores = torch.matmul(query, key.transpose(-1, -2)) / math.sqrt(d_k)
    if mask is not None:
        scores.masked_fill_(mask==0, -1e9) # padding mask
    attn = scores.softmax(-1)
    if dropout is not None:
        attn = dropout(attn)
    # the final scores: (batch, num_head, num_query, d_q)
    scores = torch.matmul(attn, value)
    return scores, attn

def make_model(src_vocab, tgt_vocab, N=6, 
               d_model=512, d_ff=2048, h=8, dropout=0.1):
    "Helper: Construct a model from hyperparameters."
    # src_vocab = 源语言词表大小
    # tgt_vocab = 目标语言词表大小
    
    c = copy.deepcopy # 对象的深度copy/clone
    attn = MultiHeadedAttention(h, d_model) # 8, 512
    # 构造一个MultiHeadAttention对象
    
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    # 512, 2048, 0.1
    # 构造一个feed forward对象

    position = PositionalEncoding(d_model, dropout)
    # 位置编码

    model = EncoderDecoder(
        Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N),
        Decoder(DecoderLayer(d_model, c(attn), c(attn), 
                             c(ff), dropout), N),
        nn.Sequential(Embeddings(d_model, src_vocab), c(position)),
        nn.Sequential(Embeddings(d_model, tgt_vocab), c(position)),
        Generator(d_model, tgt_vocab))

    # This was important from their code. 
    # Initialize parameters with Glorot / fan_avg.
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform(p)
    return model # EncoderDecoder 对象