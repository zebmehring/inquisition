import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from util import masked_softmax
from layers import HighwayEncoder





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
        self.conv1d = nn.Conv1d(in_channels=character_vectors.shape[-1], out_channels=hidden_size // 2, kernel_size=3) # not sure on kernel size
        self.proj = nn.Linear(word_vectors.size(1), hidden_size//2, bias=False)
        self.hwy = HighwayEncoder(2, hidden_size)

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
        char_emb_modified = self.conv1d(char_emb_modified.swapaxes(-1,-2)) # swap axes to get the correct shape for conv layer
        char_emb_modified = char_emb_modified.swapaxes(-1,-2) # after this, (batch_size, seq_len*word_len, hidden_size // 2)
        char_emb_modified = F.relu(char_emb_modified)

        # now, we can go back to the tensor before calling view (since they refer to the same object in memory)!
        # and take the max along word_len as desired
        # (batch_size, seq_len*word_len, hidden_dim)
        #char_emb = torch.max(char_emb, dim=2) # (batch_size, seq_len, hidden_size // 2)
        char_emb_modified = char_emb_modified.reshape(char_emb.shape[0], char_emb.shape[1], -1, char_emb.shape[-1])
        char_emb = torch.max(char_emb_modified, dim=2) # (batch_size, seq_len, hidden_size // 2)
        char_emb = F.dropout(char_emb, self.drop_prob, self.training)

        word_emb = F.dropout(word_emb, self.drop_prob, self.training)
        word_emb = self.proj(word_emb)  # (batch_size, seq_len, hidden_size // 2)

        emb = torch.cat(word_emb, char_emb, dim=2) # (batch_size, seq_len, hidden_size)

        emb = self.hwy(emb)   # (batch_size, seq_len, hidden_size)

        return emb