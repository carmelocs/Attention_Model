import torch
import torch.nn as nn

class EncoderDecoder(nn.Module):
    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator

    def forward(self, src, tgt, src_mask, tgt_mask):
        return self.decoder(self.encoder(src, src_mask), src_mask, tgt, tgt_mask)

    def encoder(self, src, src_mask):
        return self.encoder(self.src_embed(src), src_mask)

    def decoder(self, memory, src_mask, tgt, tgt_mask):
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)

class Generator(nn.Module):
    def __init__(self, d_embedding, vocab):
        # d_embedding = 512
        # vocab: tgt_vocab_size
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_embedding, vocab)

    def forward(self, x):
        return self.proj(x).softmax(-1)
        # x: (batch, num_word, d_embedding)
        # -> proj(fc): (batch, num_word, trg_vocab_size)
        # implement log_soft_max at the last dimension: (batch, num_word, trg_vocab_size)