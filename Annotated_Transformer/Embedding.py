import torch
import torch.nn as nn
import math

class Embeddings(nn.Module):
    def __init__(self, vocab, d_embedding):
        super(Embeddings, self).__init__()
        self.embedding = nn.Embedding(vocab, d_embedding)
        self.d_embedding = d_embedding

    def forward(self, x):
        x = self.embedding(x) * math.sqrt(self.d_embedding)

        return x

class PositionalEncoding(nn.Module):
    def __init__(self, d_embedding, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        self.pe = torch.zeros(max_len, d_embedding) # Position Encoding (max_num_word, d_embedding)
        self.position = torch.arange(0, max_len).unsqueeze(1)
        # unsqueeze(1) The size of tensor position must match the size of tensor div_term at non-singleton dimension 0
        self.div_term = torch.exp(torch.arange(0, d_embedding, 2) * 
                                    -(math.log(10000) / d_embedding))
        self.pe[:,0::2] = torch.sin(torch.mul(self.position, self.div_term))
        self.pe[:,1::2] = torch.cos(torch.mul(self.position, self.div_term))
        self.pe = self.pe.unsqueeze(0)
        #self.register_buffer('pe', self.pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]

        return self.dropout(x)

if __name__=='__main__':
    
    EMBEDDING_DIM = 512  # Embedding Size
    #DIM_FF = 2048 # FeedForward dimension
    #DIM_Q = DIM_K = DIM_V = 64  # dimension of K(=Q), V
    #NUM_LAYER = 6  # number of Encoder of Decoder Layer
    #NUM_HEAD = 8  # number of heads in Multi-Head Attention
    EXPECTED_MAX_NUM_WORD = 100 # ???
    VOCAB = 3200
    BATCH_SIZE = 32

    #torch.manual_seed(1)
    #input = torch.randn(BATCH_SIZE, EXPECTED_MAX_NUM_WORD).long()

    input=torch.arange(0,BATCH_SIZE*EXPECTED_MAX_NUM_WORD).view(BATCH_SIZE,EXPECTED_MAX_NUM_WORD).long()

    print('input size: {}'.format(input.shape))
    embedding = Embeddings(vocab=VOCAB, d_embedding=EMBEDDING_DIM)
    input_embedding = embedding(input)
    print('input shape after embedding: {}'.format(input_embedding.shape))
    posi = PositionalEncoding(EMBEDDING_DIM, 0.1)
    posi_encode = posi(input_embedding)
    print('input shape after posi encoding: {}'.format(posi_encode.shape))