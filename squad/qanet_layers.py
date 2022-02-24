import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from util import masked_softmax
from layers import HighwayEncoder


from torch.nn import Conv2d
from torch.nn import ModuleList
from torch.nn.functional import layer_norm
from torch.nn.functional import relu
from torch.nn.functional import softmax


import pdb


class ContextQueryAttention(torch.nn.Module):
    """Context-query attention subnetwork, as described in the QANet paper.

    See https://arxiv.org/pdf/1804.09541.pdf for more details.
    """

    def __init__(self, hidden_size, drop_prob):
        super(ContextQueryAttention, self).__init__()
        self.drop_prob = drop_prob
        self.c_weight = nn.Parameter(torch.zeros(hidden_size, 1))
        self.q_weight = nn.Parameter(torch.zeros(hidden_size, 1))
        self.cq_weight = nn.Parameter(torch.zeros(1, 1, hidden_size))
        for weight in (self.c_weight, self.q_weight, self.cq_weight):
            nn.init.xavier_uniform_(weight)
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, context, query, c_mask, q_mask):
        """
        :param context: shape (batch_size, max_content_length, hidden_size)
        :param query: shape (batch_size, max_query_length, hidden_size)
        the masks just tell you where the cs and qs are masked. (e.g., if q[1][9] -- the 9th word in the second query in this batch -- is True, then taht means that that word is masked for that query.)

 	NOTE!!! VERY VERY VERY important!!!! WE HAVE TO FIGURE OUT HOW TO DEAL WITH MASKING!! SAME FOR ENCODER BLOCK!!!
        That may actually be why the EncoderBlock isn't working!
        """
        S = self.get_similarity_matrix(context, query)# shape (batch_size, max_context_length, max-query_length)
        S_bar = softmax(S, dim=1) # shape batch_size, context_length, max_query_elngth
        S_bar_bar = softmax(S, dim=2) # shape batch_size, max_context_length, max_query_length
        A = torch.bmm(S, query) 
        B = torch.bmm(torch.bmm(S_bar, S_bar_bar.transpose(-1,-2)), context)

        output = torch.cat([context, A, context * A, context * B], dim=2)  # (batch_size, max_content_length, 4 * hid_size)
        return output

    def get_similarity_matrix(self, c, q):
        """
        Note: we got this from the BiDAF implementation.
        The reason this works is that the similarity function used by BiDAF and QANet
        is the exact same.

        See Also:
            Equation 1 in https://arxiv.org/abs/1611.01603
        """
        c_len, q_len = c.size(1), q.size(1)
        c = F.dropout(c, self.drop_prob, self.training)  # (bs, c_len, hid_size)
        q = F.dropout(q, self.drop_prob, self.training)  # (bs, q_len, hid_size)

        s0 = torch.matmul(c, self.c_weight).expand([-1, -1, q_len])
        s1 = torch.matmul(q, self.q_weight).transpose(1, 2)\
                                           .expand([-1, c_len, -1])
        s2 = torch.matmul(c * self.cq_weight, q.transpose(1, 2))
        s = s0 + s1 + s2 + self.bias

        return s




class SelfAttention(torch.nn.Module):
    """
    SelfAttention as described in Attention is All You Need.
    """
    def __init__(self, hidden_size, num_attn_heads):
        """

        :param hidden_size (int): hidden size of input
        :param num_attn_heads (int): the number of attention heads
        """
        super(SelfAttention, self).__init__()
        self.num_attn_heads = num_attn_heads
        self.hidden_size = hidden_size

        self.Qs = ModuleList([torch.nn.Linear(
            in_features=hidden_size, out_features=hidden_size, bias=False)
        for _ in range(num_attn_heads)])
        self.Ks = ModuleList([torch.nn.Linear(
            in_features=hidden_size, out_features=hidden_size, bias=False)
                              for _ in range(num_attn_heads)])
        self.Vs = ModuleList([torch.nn.Linear(
            in_features=hidden_size, out_features=hidden_size, bias=False)
                              for _ in range(num_attn_heads)])

        self.proj = nn.Linear(in_features = num_attn_heads * hidden_size,
                              out_features = hidden_size, bias=False)


    def forward(self, x, mask=None):
        """

        :param x: has shape (batch_size, seq_len, hidden_size)
	:param mask: tensor with shape (batch_size, seq_len) 
        :return:
        """
        #pdb.set_trace()

        attention_outputs = None
        for i in range(self.num_attn_heads):
            logits = torch.bmm(self.Qs[i](x), self.Ks[i](x).transpose(-1, -2) / torch.sqrt(torch.tensor(self.hidden_size, dtype=torch.float32)))
            attention_scores = None
            if mask is None:
                attention_scores = softmax(logits)
            else:
                attention_scores = masked_softmax(logits, torch.bmm(mask.long().unsqueeze(2), mask.long().unsqueeze(1))) 
            output = torch.bmm(attention_scores, self.Vs[i](x))
            if attention_outputs is None:
                attention_outputs = output
            else:
                attention_outputs = torch.cat((attention_outputs, output), dim=-1)

        output = self.proj(attention_outputs)

        return output


class EncoderBlock(torch.nn.Module):
    """Encoder block subnetwork, as described in the QANet paper.

    Accepts an input embedding and transforms it according to the following
    sequence of operations:
      - positional encoding
      - convolution
      - self-attention
      - non-linear, feed-forward transformation
    The parameters controlling these transformations are specified by the
    arguments to `__init__`.

    See https://arxiv.org/pdf/1804.09541.pdf for more details.
    """

    def __init__(self, hidden_size=128, num_convs=4, num_attn_heads=8):
        """Constructs an encoder block module.

        Args:
          hidden_size [int]: the size of the feature representations used by the model;
            also the output size
          num_convs [int]: the number of convolutions to perform
          num_attn_heads [int]: the number of heads to use in the self-attention layer
        """
        super(EncoderBlock, self).__init__()

        self.position_encoder = PositionEncoder(hidden_size)
        
        self.hidden_size = hidden_size

        # Note: we set padding=3 below to maintain the dimensionality of the input
	# We get this using the equation for L_out given in conv1d documentation
	# othewise, each conv operation would reduce dimensionality of the input, which is probably not desirable
	# since we would no longer have one vector per index in the sequence.
        self.convs = ModuleList([nn.Conv1d(in_channels=hidden_size, out_channels=hidden_size, kernel_size=7, padding=3)
                                 for _ in range(num_convs)])
        
        self.layer_norm = nn.LayerNorm(normalized_shape=hidden_size)

        self.att = SelfAttention(hidden_size, num_attn_heads)


        self.ff = torch.nn.Linear(
            in_features=hidden_size, out_features=hidden_size, bias=True)
	# there's another weight plus bias multipliation. (See page 5 of Attention is All You Need).
        self.ff2 = torch.nn.Linear(in_features=hidden_size, out_features=hidden_size, bias=True)

    def forward(self, x, x_mask):
        """
        :param x: tensor with shape (batch_size, seq_len, hidden_size)
        :param x_mask: tensor with shape (batch_size, seq_len)  for which x_mask[i][j] = False if jth character of ith sequence is masked and True otherwise. We'll use this to zero out and get negative infinity where necessary.
        """
        # this just extends out the x_mask to have shape (batch_size, seq_len_ hiddensize)
	# basically, extended_mask[i][j][k] = x_mask[i][j] for all 0 \leq k < x.shape[-1]
        extended_mask = x_mask.unsqueeze(2).repeat(1,1,x.shape[-1])
    
        
        output = self.position_encoder(x) # (batch_size, seq_len, hidden_size)

        for conv in self.convs:
            residual = output
            output = self.layer_norm(output)
            output = conv(output.transpose(-1,-2)) # by transposing it, we get (batch_size, hidden_size, seq_len). Looking at the conv1d docs, this makes our in_channels equal to hidden_size as desired.
            output = output.transpose(-1,-2) # now, just tranpoase it back to (batch_size, seq_len, hidden_size)

        # zero out the masked output tokens
        #residual = (output * x_mask.reshape(x_mask.shape[0], x_mask.shape[1], 1)) # zeros out the vectors corresponding to masked tokens
        #zero_masked_output = torch.clone(output)
        #zero_masked_output[~extended_mask] = 0

        #residual = zero_masked_output 
        residual = output
        output = self.layer_norm(output) # (batch_size, seq_len, hidden_size)

        # recall that self.Q(output) has shape (batch_size, seq_len. hidden_size) and same for self.K(output) and V(outupt)
	# for this reason, we ahve to use bmm instead of regular matrix multplication and we also have to transpose
	# the non-batch dimensions
        #pdb.set_trace()
        """
        negative_inf_mask = torch.ones((output.shape[0], output.shape[1]))
        negative_inf_mask[x_mask == False] = float('-inf') 
        output = self.att(output * negative_inf_mask.reshape(output.shape[0], output.shape[1], 1))
        """
        #neg_inf_masked_output = torch.clone(output)
        #neg_inf_masked_output[~extended_mask] = float('-inf')

        output = self.att(output, x_mask)
        #pdb.set_trace()
        output += residual

        residual = output
        output = self.layer_norm(output)
        output = relu(self.ff(output))
        output = self.ff2(output)
        output += residual

        return output



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
