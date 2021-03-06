"""Top-level model classes.

Author:
    Chris Chute (chute@stanford.edu)
"""

import layers
import qanet_layers
import torch
import torch.nn as nn

import pdb


class QANet(nn.Module):
    """QANet model for SQuAD.


    Follows the high-level structure:
        - Embedding layer: TODO
        - Encoder layer: NEED TO TEST
        - Attention layer: TODO
        - Model encoder layer: NEED TO TEST
        - Output layer: TODO

    Args:
        word_vectors (torch.Tensor): Pre-trained word vectors.
        character_vectors (torch.Tensor): Pre-trained character vectors
        hidden_size (int): Number of features in the hidden state at each layer.
        drop_prob (float): Dropout probability.
        num_enc_blocks(list[int]): a two element list giving the number of times to apply the encoder blocks
					for the embedding encoder layer and model encoder layer, respectively
    """
    def __init__(self, word_vectors, character_vectors, hidden_size, device, drop_prob, num_enc_blocks=[1,5]):
        super(QANet, self).__init__()

        self.emb = qanet_layers.Embedding(word_vectors=word_vectors,
                                    character_vectors = character_vectors,
                                    hidden_size=hidden_size,
                                    drop_prob=drop_prob)

        self.emb_encs = nn.ModuleList([qanet_layers.EncoderBlock(hidden_size = hidden_size, device=device, drop_prob = drop_prob, num_convs=4, num_attn_heads=8, kernel_size=7) for _ in range(num_enc_blocks[0])]) 

        self.att = layers.BiDAFAttention(hidden_size=hidden_size, drop_prob = drop_prob)

        self.mod_encs = nn.ModuleList([qanet_layers.EncoderBlock(hidden_size=4*hidden_size, device=device, drop_prob = drop_prob, num_convs=2, num_attn_heads=8, kernel_size=5) for _ in range(num_enc_blocks[1])])

        self.out = qanet_layers.OutputLayer(hidden_size = hidden_size)


    def forward(self, cw_idxs, cc_idxs, qw_idxs, qc_idxs):

        # note: we don't need to change the masking below.
        # the rason is that that masking is used for the attentino computation.
        # the character level embeddings have nothing to do with the attention computation,
        # do we don't need to adjust the masking.
        c_mask = torch.zeros_like(cw_idxs) != cw_idxs
        q_mask = torch.zeros_like(qw_idxs) != qw_idxs
        c_len, q_len = c_mask.sum(-1), q_mask.sum(-1)

        c_emb = self.emb((cw_idxs, cc_idxs))         # (batch_size, c_len, hidden_size)
        q_emb = self.emb((qw_idxs, qc_idxs))         # (batch_size, q_len, hidden_size)

        # pdb.set_trace()
	# NOTE: we need to account for padding somehow!!!
	# I'm going to ask in office hours about this. It's not clear to me how to do this.
	# since we're appling convoluations first, not sure
        #pdb.set_trace()

        

        #c_enc = self.enc(c_emb, c_len)
        #q_enc = self.enc(q_emb, q_len)

        c_enc = c_emb 
        q_enc = q_emb 
        for i in range(len(self.emb_encs)):
            c_enc = self.emb_encs[i](c_enc, c_mask)    # (batch_size, max_context_len, hidden_size)
            q_enc = self.emb_encs[i](q_enc, q_mask)    # (batch_size, max_query_len, hidden_size)



        att = self.att(c_enc, q_enc,
                       c_mask, q_mask)    # (batch_size, c_len, 4 * hidden_size)

        mods = list()

        # there are three layers of encoder block stacks.
        # each shares parameters (i.e. is the same block stack)
        # so we just run our output through this three times and save some intermediate outputs.
        mod = att 
        for _ in range(3):
            for i in range(len(self.mod_encs)):
                mod = self.mod_encs[i](mod, c_mask)        # (batch_size, c_len, 4 * hidden_size)
            mods.append(mod)
        #mod = self.mod(att, c_mask)

        #out = self.out(att, mod, c_mask)  # 2 tensors, each (batch_size, c_len)
        out = self.out(mods, c_mask)

        return out



class BiDAF(nn.Module):
    """Baseline BiDAF model for SQuAD.

    Based on the paper:
    "Bidirectional Attention Flow for Machine Comprehension"
    by Minjoon Seo, Aniruddha Kembhavi, Ali Farhadi, Hannaneh Hajishirzi
    (https://arxiv.org/abs/1611.01603).

    Follows a high-level structure commonly found in SQuAD models:
        - Embedding layer: Embed word indices to get word vectors.
        - Encoder layer: Encode the embedded sequence.
        - Attention layer: Apply an attention mechanism to the encoded sequence.
        - Model encoder layer: Encode the sequence again.
        - Output layer: Simple layer (e.g., fc + softmax) to get final outputs.

    Args:
        word_vectors (torch.Tensor): Pre-trained word vectors.
        hidden_size (int): Number of features in the hidden state at each layer.
        drop_prob (float): Dropout probability.
    """
    def __init__(self, word_vectors, hidden_size, drop_prob=0.):
        super(BiDAF, self).__init__()
        self.emb = layers.Embedding(word_vectors=word_vectors,
                                    hidden_size=hidden_size,
                                    drop_prob=drop_prob)

        self.enc = layers.RNNEncoder(input_size=hidden_size,
                                     hidden_size=hidden_size,
                                     num_layers=1,
                                     drop_prob=drop_prob)

        self.att = layers.BiDAFAttention(hidden_size=2 * hidden_size,
                                         drop_prob=drop_prob)

        self.mod = layers.RNNEncoder(input_size=8 * hidden_size,
                                     hidden_size=hidden_size,
                                     num_layers=2,
                                     drop_prob=drop_prob)

        self.out = layers.BiDAFOutput(hidden_size=hidden_size,
                                      drop_prob=drop_prob)

    def forward(self, cw_idxs, qw_idxs):
        c_mask = torch.zeros_like(cw_idxs) != cw_idxs
        q_mask = torch.zeros_like(qw_idxs) != qw_idxs
        c_len, q_len = c_mask.sum(-1), q_mask.sum(-1)

        c_emb = self.emb(cw_idxs)         # (batch_size, c_len, hidden_size)
        q_emb = self.emb(qw_idxs)         # (batch_size, q_len, hidden_size)
   

        c_enc = self.enc(c_emb, c_len)    # (batch_size, c_len, 2 * hidden_size)
        q_enc = self.enc(q_emb, q_len)    # (batch_size, q_len, 2 * hidden_size)

        att = self.att(c_enc, q_enc,
                       c_mask, q_mask)    # (batch_size, c_len, 8 * hidden_size)

        mod = self.mod(att, c_len)        # (batch_size, c_len, 2 * hidden_size)

        out = self.out(att, mod, c_mask)  # 2 tensors, each (batch_size, c_len)

        return out
