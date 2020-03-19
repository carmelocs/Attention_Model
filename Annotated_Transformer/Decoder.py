import torch
import torch.nn as nn
from MultiHeadAttention import MultiHeadAttention
from PositionwiseFeedForward import PositionwiseFeedForward
from utils import clones
from Encoder import SublayerConnection

class DecoderLayer(nn.Module):
    "A DecoderLayer is made of self-attn, src-attn and feed forward"
    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        # size = d_embedding = 512,
        # self_attn: one MultiHeadAttention object between tgt_vocab
        # src_attn: second MultiHeadAttention object, betweem tgt_vocab and src_vocab
        # feed_forward: the last fully-connection layer
        # dropout = 0.1
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        # Three SublayerConnection objects: 
        # self.self_attn, self.src_attn, and self.feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 3)

    def forward(self, x, memory, src_mask, tgt_mask):
        m = memory
        # memory: (batch, num_word, d_embedding)
        # the output of src_vocab Encoding, works as key and value for tgt_vocab query
        x = self.sublayer[0](x, self.self_attn(x, x, x, tgt_mask))
        # to compute the self_attention encoding scores for each query in tgt_vocab
        x = self.sublayer[1](x, self.src_attn(x, m, m, src_mask))
        # to compute attention scores between query from tgt_vocab and {key, value} in src_vocab
        return self.sublayer[2](x, self.feed_forward)


class Decoder(nn.Module):
    def __init__(self, layer, N):
        # layer: a DecoderLayer object, N = 6
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = nn.LayerNorm(layer.size)
        # initialize a nn.LayerNorm

    def forward(self, x, memory, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
            # implement six times DecoderLayer
        return self.norm(x)
        # apply LayerNorm after six DecoderLayers: (batch, num_word, d_embedding)