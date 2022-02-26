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


class OutputLayer(torch.nn.Module):
    """
    """

    def __init__(self, hidden_size):
        '''

        :param hidden_size: hidden size of all layers in the QANet network.
        '''
        super(OutputLayer, self).__init__()

        # 8 times hidden_size because each layer multilies two outputs fro mteh model encoder block
        # and input to model encoder blocks will have last dim shape equal to 4*hidden_size since the input
        # input to model encoder blocks cmoes from c2q attention (and that's how big the output of that function is)
        self.start_span_linear = nn.Linear(8*hidden_size, 1, bias=False)
        self.end_span_linear = nn.Linear(8*hidden_size, 1, bias=False)

    def forward(self, model_encoder_outputs, mask):
        '''
        :param model_encoder_outputs (list[torch.tensor]): a list of model encoding outputs
        This is M0, M1, M2 fro mthe paper (in that order).
        :param mask
        '''
        # both of the logits have shape (batch_size, seq_len, 1)
        logits_start = self.start_span_linear(torch.cat((model_encoder_outputs[0], model_encoder_outputs[1]), dim=-1))
        logits_end = self.start_span_linear(torch.cat((model_encoder_outputs[0], model_encoder_outputs[2]), dim=-1))

        # After squeezing, both logits have shape (batch_size, seq_len)
        log_p1 = masked_softmax(logits_start.squeeze(), mask, log_softmax=True)
        log_p2 = masked_softmax(logits_end.squeeze(), mask, log_softmax=True)

        return log_p1, log_p2


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

    def forward(self, c, q, c_mask, q_mask):
        batch_size, c_len, _ = c.size()
        q_len = q.size(1)
        s = self.get_similarity_matrix(c, q)  # (batch_size, c_len, q_len)
        c_mask = c_mask.view(batch_size, c_len, 1)  # (batch_size, c_len, 1)
        q_mask = q_mask.view(batch_size, 1, q_len)  # (batch_size, 1, q_len)
        s1 = masked_softmax(s, q_mask, dim=1)  # (batch_size, c_len, q_len)
        s2 = masked_softmax(s, c_mask, dim=2)  # (batch_size, c_len, q_len)

        # (bs, c_len, q_len) x (bs, q_len, hid_size) => (bs, c_len, hid_size)
        a = torch.bmm(s1, q)
        # (bs, c_len, c_len) x (bs, c_len, hid_size) => (bs, c_len, hid_size)
        b = torch.bmm(torch.bmm(s1, s2.transpose(1, 2)), c)

        x = torch.cat([c, a, c * a, c * b], dim=2)  # (bs, c_len, 4 * hid_size)

        return x

    def get_similarity_matrix(self, c, q):
        """Get the "similarity matrix" between context and query (using the
        terminology of the BiDAF paper).

        A naive implementation as described in BiDAF would concatenate the
        three vectors then project the result with a single weight matrix. This
        method is a more memory-efficient implementation of the same operation.

        See Also:
            Equation 1 in https://arxiv.org/abs/1611.01603
        """
        c_len, q_len = c.size(1), q.size(1)
        c = F.dropout(c, self.drop_prob, self.training)  # (bs, c_len, hid_size)
        q = F.dropout(q, self.drop_prob, self.training)  # (bs, q_len, hid_size)

        # pdb.set_trace()
        # Note to self: the BIDAF model is epxecting c \in (batch_size, seq_len, 2*hidden_size), where hidden_size
        # is the hidden_size passed to the ender.
        # the reason is that they do forward and backward RNN encoding and then concatenate the two final hidden layers
        # and so their encoder outputs something that is 2 times the hidden_size in that case
        # QANet does not do this.
        # Shapes: (batch_size, c_len, q_len)
        s0 = torch.matmul(q, self.q_weight).expand([-1, -1, c_len])
        s1 = torch.matmul(c, self.c_weight).transpose(1, 2) \
            .expand([-1, q_len, -1])
        s2 = torch.matmul(q * self.cq_weight, c.transpose(1, 2))
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


        self.w_q = nn.Linear(in_features=hidden_size, out_features=hidden_size, bias=False)
        self.w_k = nn.Linear(in_features=hidden_size, out_features=hidden_size, bias=False)
        self.w_v = nn.Linear(in_features=hidden_size, out_features=hidden_size, bias=False)

        self.projection = nn.Linear(in_features=hidden_size, out_features=hidden_size, bias=False)


    def forward(self, x, mask=None):
        """

        :param x: has shape (batch_size, seq_len, hidden_size)
	:param mask: tensor with shape (batch_size, seq_len) 
        :return:
        """
        # these will have same dimensions as x
        Q = self.w_q(x)
        K = self.w_k(x)
        V = self.w_v(x)

        # get the Q, K, and V for each separate attention head.
        # these will be (batch_size, seq_len, num_attn_heads, hidden_size // num_attn_heads)
        Q = Q.reshape(x.shape[0], x.shape[1], self.num_attn_heads, x.shape[-1] // self.num_attn_heads)
        K = K.reshape(x.shape[0], x.shape[1], self.num_attn_heads, x.shape[-1] // self.num_attn_heads)
        V = V.reshape(x.shape[0], x.shape[1], self.num_attn_heads, x.shape[-1] // self.num_attn_heads)

        # ensure that we can get the attention values per-head through matrix multiplication
        # (because now QK^T will give us (batch_size, num_attn_heads, seq_len, seq_len), which will be the
        # attention scores per head, where each head pays attention to a different subset of the x hidden size dimensions)
        # these will have shape (batch_size, num_attn_heads, seq_len, hidden_size // num_attn_heads)
        Q = Q.permute(0, 2, 1, 3)
        K = K.permute(0, 2, 1, 3)
        V = V.permute(0, 2, 1, 3)

        # get the attention scores
        # shape (batch_size, num_attn_heads, seq_len, seq_len)
        logits = torch.matmul(Q, K.transpose(-1, -2)) / torch.sqrt(torch.tensor(self.hidden_size // self.num_attn_heads, dtype=torch.float32))
        mask = mask.float().unsqueeze(1).repeat(1,self.num_attn_heads,1) # shape (batch_size, num_attn_heads, seq_len)
        mask = torch.matmul(mask.unsqueeze(-1), mask.unsqueeze(-2)) # shape (batch_size, num_attn_heads, seq_len, seq_len). The masked out quantities will be rows and columns corresponding to padded tokens.
        attention_scores = masked_softmax(logits, mask)

        # get the attention vectors
        output = torch.matmul(attention_scores, V) # shape (batch_size, num_attn_heads, seq_len, hidden_size // num_attn_heads)
        output = output.permute(0, 2, 1, 3) # shape (batch_size, seq_len, num_attn_heads, hidden_size // num_attn_heads)
        output = output.view(output.shape[0], output.shape[1], self.hidden_size) # shape (batch_size, seq_len, hidden_size)

        # I think we only do the projection if we're doing multiheaded attention, right?
        if self.num_attn_heads != 1:
            output = self.projection(output)

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
     
        self.num_convs = num_convs 
        
        self.hidden_size = hidden_size

        # Note: we set padding=3 below to maintain the dimensionality of the input
	# We get this using the equation for L_out given in conv1d documentation
	# othewise, each conv operation would reduce dimensionality of the input, which is probably not desirable
	# since we would no longer have one vector per index in the sequence.
        self.convs = ModuleList([nn.Conv1d(in_channels=hidden_size, out_channels=hidden_size, kernel_size=7, padding=3)
                                 for _ in range(num_convs)])
        
        self.layer_norms = ModuleList([nn.LayerNorm(normalized_shape=hidden_size) for _ in range(num_convs+2)])

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
        
        output = self.position_encoder(x) # (batch_size, seq_len, hidden_size)

        for i, conv in enumerate(self.convs):
            residual = output
            output = self.layer_norms[i](output)
            output = conv(output.transpose(-1,-2)) # by transposing it, we get (batch_size, hidden_size, seq_len). Looking at the conv1d docs, this makes our in_channels equal to hidden_size as desired.
            output = output.transpose(-1,-2) # now, just tranpoase it back to (batch_size, seq_len, hidden_size)

        residual = output
        output = self.layer_norms[self.num_convs](output) # (batch_size, seq_len, hidden_size)

        output = self.att(output, x_mask)
        output += residual

        residual = output
        output = self.layer_norms[self.num_convs+1](output)
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
