import torch
import torch.nn as nn
from MultiHeadAttention import MultiHeadAttention
from PositionwiseFeedForward import PositionwiseFeedForward
from utils import clones

class Encoder(nn.Module):
    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = nn.LayerNorm(layer.size)
        
    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class SublayerConnection(nn.Module):
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = nn.LayerNorm(size)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))


class EncoderLayer(nn.Module):
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size
    
    def forward(self, x, mask):
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)


if __name__ == '__main__':
    EMBEDDING_DIM = 512  # Embedding Size
    DIM_FF = 2048 # FeedForward dimension
    DIM_Q = DIM_K = DIM_V = 64  # dimension of K(=Q), V
    NUM_LAYER = 6  # number of Encoder and Decoder Layer
    NUM_HEAD = 8  # number of heads in Multi-Head Attention
    EXPECTED_MAX_NUM_WORD = 100 # ???
    BATCH_SIZE = 32

    attn = MultiHeadAttention(head=NUM_HEAD, d_embedding=EMBEDDING_DIM)
    feed_forward = PositionwiseFeedForward(d_embedding=EMBEDDING_DIM, d_ff=DIM_FF)
    encoder_layer = EncoderLayer(size=EMBEDDING_DIM, self_attn=attn, feed_forward=feed_forward, dropout=0.1)
    encoder = Encoder(layer=encoder_layer, N=6)
    print(encoder_layer)