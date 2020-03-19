import torch
import torch.nn as nn
import torch.nn.functional as F

class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_embedding, d_ff, dropout=0.1):
        # d_embedding = 512, d_ff = 2048
        super(PositionwiseFeedForward, self).__init__()
        # Construct the first fully connection layer, 
        # weights: (d_embedding, d_ff), biases: (d_ff)
        self.w_1 = nn.Linear(d_embedding, d_ff)
        # Construct the second fully connection layer, 
        # weights: (d_ff, d_embedding), biases: (d_embedding)
        self.w_2 = nn.Linear(d_ff, d_embedding)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        x = self.w_1(x)
        #print('shape after first linear layer: {}'.format(x.shape))

        return self.w_2(self.dropout(self.relu(x)))
        # x (batch, num_word, d_embedding) -> self.w_1 -> (batch, num_word, d_ff)
        # -> relu -> (batch, num_word, d_ff)
        # -> dropout -> (batch, num_word, d_ff)
        # -> self.w_2 -> (batch, num_word, d_embedding)

if __name__=='__main__':
    EMBEDDING_DIM = 512  # Embedding Size
    DIM_FF = 2048 # FeedForward hidden layer dimension
    EXPECTED_MAX_NUM_WORD = 100 # ???
    BATCH_SIZE = 32

    input = torch.empty([BATCH_SIZE, EXPECTED_MAX_NUM_WORD, EMBEDDING_DIM])
    posi = PositionwiseFeedForward(d_embedding=EMBEDDING_DIM, d_ff=DIM_FF)
    output = posi(input)
    print('shape after PositionwiseFeedForward: {}'.format(output.shape))