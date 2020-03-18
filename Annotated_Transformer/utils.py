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