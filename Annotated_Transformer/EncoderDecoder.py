import torch
import torch.nn as nn

class EncoderDecoder(nn.Module):
    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        # an Encoder object
        self.decoder = decoder
        # a Decoder object
        self.src_embed = src_embed
        # src_vocab embedding(including word-to-vector embedding and positional encoding)
        self.tgt_embed = tgt_embed
        # tgt_vocab embedding(including word-to-vector embedding and positional encoding)
        self.generator = generator
        # a Generator object

    def forward(self, src, tgt, src_mask, tgt_mask):
        return self.decoder(self.encoder(src, src_mask), src_mask, tgt, tgt_mask)
        #  先对源语言序列进行编码，
        # 结果作为memory传递给目标语言的编码器

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