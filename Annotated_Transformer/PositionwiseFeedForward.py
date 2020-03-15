import torch
import torch.nn as nn
import torch.nn.functional as F

class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_embedding, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_embedding, d_ff)
        self.w_2 = nn.Linear(d_ff, d_embedding)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        x = self.w_1(x)
        #print('shape after first linear layer: {}'.format(x.shape))
        x = self.dropout(self.relu(x))
        x = self.w_2(x)

        return x

if __name__=='__main__':
    EMBEDDING_DIM = 512  # Embedding Size
    DIM_FF = 2048 # FeedForward dimension
    EXPECTED_MAX_NUM_WORD = 100 # ???
    BATCH_SIZE = 32

    input = torch.empty([BATCH_SIZE, EXPECTED_MAX_NUM_WORD, EMBEDDING_DIM])
    posi = PositionwiseFeedForward(d_embedding=EMBEDDING_DIM, d_ff=DIM_FF)
    output = posi(input)
    print('shape after PositionwiseFeedForward: {}'.format(output.shape))