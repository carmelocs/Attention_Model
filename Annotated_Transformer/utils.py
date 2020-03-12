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
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-1, -2)) / math.sqrt(d_k)

    if mask is not None:
        scores.masked_fill_(mask==0, -1e9)

    attn = scores.softmax(-1)

    if dropout is not None:
        attn = dropout(attn)

    scores = torch.matmul(attn, value)

    return scores, attn
    