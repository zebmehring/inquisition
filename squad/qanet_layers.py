import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from util import masked_softmax
from layers import HighwayEncoder

import pdb



class PositionEncoder(torch.nn.Module):
    def __init__(self, hidden_size, max_seq_len=1000):
        """

        :param hidden_size: the dimensionality of the input
        :param max_seq_len (int): the maximum sequence length that needs to be positionally
         encoded. I set it 1000 because this value will really be the maximum of
         args.quest_limit, args.para_limit, args.test_quest_limit, args.test_para_limit
         (since those are sequences which will be encoded: the question and context vectors)

         We want to compute the position encoding p_t for all t \in 0, ..., max_seq_len.
         Note that each p_t has shape (hidden_size)
         and so position_encodings will have shape (max_seq_len, hidden_size)
        """
        super(PositionEncoder, self).__init__()


        freq_indices = torch.arange(hidden_size//2)#.repeat_interleave(2)
        frequencies = torch.pow(10000, (2*freq_indices)/hidden_size)


        #frequencies = torch.pow(10000, 2 * ((hidden_size // 2) / )

        positions = torch.arange(max_seq_len).reshape(-1, 1)
        positions = positions.repeat(1, hidden_size // 2) # shape (max_seq_len, hidden-size // 2). These are the t values in p_t


        self.position_encodings = torch.zeros((max_seq_len, hidden_size))


        self.position_encodings[:, 0::2] = torch.sin(positions / frequencies)
        self.position_encodings[:, 1::2] = torch.cos(positions / frequencies)

    def forward(self, x):
        """

        :param x: tensor with shape (batch_size, seq_len, hidden_size)
        :return:
        """
        # note that we only get the first seq_len position encodings (since max_seq_len
        # may be greater than seq_len)
        return self.position_encodings[:x.shape[1]] + x



class Embedding(nn.Module):
    """Embedding layer used by QANet

    Word-level embeddings are further refined using a 2-layer Highway Encoder
    (see `HighwayEncoder` class for details).

    Args:
        word_vectors (torch.Tensor): Pre-trained word vectors.
        hidden_size (int): Size of hidden activations.
        drop_prob (float): Probability of zero-ing out activations
    """
    def __init__(self, word_vectors, character_vectors, hidden_size, drop_prob):
        super(Embedding, self).__init__()
        self.drop_prob = drop_prob
        self.char_embed = nn.Embedding.from_pretrained(character_vectors)
        self.word_embed = nn.Embedding.from_pretrained(word_vectors)
        self.conv1d = nn.Conv1d(in_channels=character_vectors.shape[-1], out_channels=hidden_size // 2, kernel_size=3, padding=1) # not sure on kernel size
        self.proj = nn.Linear(word_vectors.size(1), hidden_size//2, bias=False)
        self.hwy = HighwayEncoder(2, hidden_size)

        # self.test = PositionEncoder(hidden_size)

    def forward(self, x):
        """

        :param x: A tuple. First element is word_idxs. Second is character_idxs.
         TODO: get the shape of these and report that here.
         I think that:
         word_idxs has shape (batch_size, seq_len)
         char_idxs has shape (batch_size, seq_len, word_len)
         # not sure if the word_len thing will be constant, but assume it's some constant length (word_len)
        :return:
        """
        word_idxs, char_idxs = x
        word_emb = self.word_embed(word_idxs)   # (batch_size, seq_len, word_embed_size)
        char_emb = self.char_embed(char_idxs) # (batch_size, seq_len, word_len, char_embed_size)
        char_emb_modified = char_emb.view(char_emb.shape[0], char_emb.shape[1]*char_emb.shape[2], char_emb.shape[-1]) # collapse it so that all char embeddings are in same axis; no longer split by word index
        # (batch_size, seq_len*word_len, char_embed_size)

        char_emb_modified = self.conv1d(torch.transpose(char_emb_modified, -1,-2)) # swap axes to get the correct shape for conv layer
        char_emb_modified = torch.transpose(char_emb_modified, -1,-2) # after this, (batch_size, seq_len*word_len, hidden_size // 2)
        char_emb_modified = F.relu(char_emb_modified) 

        # now, we can go back to the tensor before calling view (since they refer to the same object in memory)!
        # and take the max along word_len as desired
        # (batch_size, seq_len*word_len, hidden_dim)
        #char_emb = torch.max(char_emb, dim=2) # (batch_size, seq_len, hidden_size // 2)
        char_emb_modified = char_emb_modified.reshape(char_emb.shape[0], char_emb.shape[1], char_emb.shape[2], char_emb_modified.shape[-1])
        char_emb = torch.max(char_emb_modified, dim=2)[0] # (batch_size, seq_len, hidden_size // 2)
        char_emb = F.dropout(char_emb, self.drop_prob, self.training)

        word_emb = F.dropout(word_emb, self.drop_prob, self.training)
        word_emb = self.proj(word_emb)  # (batch_size, seq_len, hidden_size // 2)

        emb = torch.cat((word_emb, char_emb), dim=2) # (batch_size, seq_len, hidden_size)

        emb = self.hwy(emb)   # (batch_size, seq_len, hidden_size)
  
        # emb = self.test(emb)

        return emb
