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


class StochasticDepth(torch.nn.Module):
    """ 
    """
    def __init__(self, l, L, p_L=0.9):
        super(StochasticDepth, self).__init__()
   
        self.p_l = 1 - (1 / L) * (1- p_L)
        self.p_l_tensor = torch.tensor([self.p_l], dtype=torch.float32) 

    def forward(self, x):
        # at training time, dropout the input 
        if self.training:
            keep = torch.bernoulli(self.p_l_tensor) 
            return keep * x 
        # at test time, we just multiply the output by p_keep
        return self.p_l * x 

class DepthwiseSeparableConv1d(torch.nn.Module):
    """ Got help from https://www.paepper.com/blog/posts/depthwise-separable-convolutions-in-pytorch/. Still don't fully understands the 'groups' parameter """
    def __init__(self, in_channels, out_channels, kernel_size, padding):
        super(DepthwiseSeparableConv1d, self).__init__()
 
        self.depthwise_conv = nn.Conv1d(in_channels=in_channels, out_channels=in_channels, kernel_size=kernel_size, padding = padding, groups=in_channels)
        self.pointwise_conv = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, padding=0)

    def forward(self, x):
        return self.pointwise_conv(self.depthwise_conv(x))


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

        # MAYBE PROJECT BEFORE INPUTTING TO MHA!

        # TEMPORARY!!! FOR TESTING
        self.att = nn.MultiheadAttention(hidden_size, num_attn_heads, bias=False)
        
        self.w_q = nn.Linear(in_features=hidden_size, out_features=hidden_size, bias=False)
        self.w_k = nn.Linear(in_features=hidden_size, out_features=hidden_size, bias=False)
        self.w_v = nn.Linear(in_features=hidden_size, out_features=hidden_size, bias=False)

        """
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

        """

    def forward(self, x, mask=None):
        """

        :param x: has shape (batch_size, seq_len, hidden_size)
	:param mask: tensor with shape (batch_size, seq_len) 
        :return:
        """

        """
        attention_outputs = None
        for i in range(self.num_attn_heads):
            logits = torch.bmm(self.Qs[i](x), self.Ks[i](x).transpose(-1, -2) / torch.sqrt(torch.tensor(self.hidden_size, dtype=torch.float32)))
            attention_scores = None
            if mask is None:
                attention_scores = softmax(logits)
            else:
                attention_scores = masked_softmax(logits, torch.bmm(mask.float().unsqueeze(2), mask.float().unsqueeze(1))) 
            output = torch.bmm(attention_scores, self.Vs[i](x))
            if attention_outputs is None:
                attention_outputs = output
            else:
                attention_outputs = torch.cat((attention_outputs, output), dim=-1)

        output = self.proj(attention_outputs)

        return output
        """
        x = torch.transpose(x, 0, 1)
        #temp_x = temp_x.repeat(1,1, self.num_attn_heads)
        x = self.att(self.w_q(x), self.w_k(x), self.w_v(x), key_padding_mask = ~mask)[0]
        return torch.transpose(x, 0, 1)


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

    def __init__(self, hidden_size, device, num_convs, num_attn_heads, kernel_size, drop_prob):
        """Constructs an encoder block module.

        Args:
          hidden_size [int]: the size of the feature representations used by the model;
            also the output size
          num_convs [int]: the number of convolutions to perform
          num_attn_heads [int]: the number of heads to use in the self-attention layer
        """
        super(EncoderBlock, self).__init__()

        self.num_convs = num_convs 
        self.drop_prob = drop_prob
        self.position_encoder = PositionEncoder(hidden_size, device, drop_prob)
        
        self.hidden_size = hidden_size

        # Note: we set padding=3 below to maintain the dimensionality of the input
	# We get this using the equation for L_out given in conv1d documentation
	# othewise, each conv operation would reduce dimensionality of the input, which is probably not desirable
	# since we would no longer have one vector per index in the sequence.
        self.convs = ModuleList([DepthwiseSeparableConv1d(in_channels=hidden_size, out_channels=hidden_size, kernel_size=kernel_size, padding=(kernel_size - 1) // 2) for _ in range(num_convs)])
        
        self.layer_norms = ModuleList([nn.LayerNorm(normalized_shape=hidden_size) for _ in range(num_convs+2)])

        self.att = SelfAttention(hidden_size, num_attn_heads)


        # TA said to use kenrel_size = 1 and padding = 0 for these
        self.ff = nn.Conv1d(in_channels = hidden_size, out_channels=hidden_size, kernel_size=1, padding=0) 
        self.ff2 = nn.Conv1d(in_channels = hidden_size, out_channels=hidden_size, kernel_size=1, padding=0) 

    def forward(self, x, x_mask):
        """
        :param x: tensor with shape (batch_size, seq_len, hidden_size)
        :param x_mask: tensor with shape (batch_size, seq_len)  for which x_mask[i][j] = False if jth character of ith sequence is masked and True otherwise. We'll use this to zero out and get negative infinity where necessary.
        """
        
        output = self.position_encoder(x) # (batch_size, seq_len, hidden_size)

        for i, conv in enumerate(self.convs):
            residual = output 
            output = self.layer_norms[i](output)
            output = F.dropout(output, self.drop_prob, self.training)
            output = conv(output.transpose(-1,-2)).transpose(-1,-2)# by transposing it, we get (batch_size, hidden_size, seq_len). Looking at the conv1d docs, this makes our in_channels equal to hidden_size as desired.
            output = F.relu(output)
            output = output + residual

        residual = output 
        output = self.layer_norms[self.num_convs](output) # (batch_size, seq_len, hidden_size)
        output = F.dropout(output, self.drop_prob, self.training)
        output = self.att(output, x_mask)
        output = output + residual

        residual = output 
        output = self.layer_norms[self.num_convs+1](output)
        output = F.dropout(output, self.drop_prob, self.training)
        output = relu(self.ff(output.transpose(-1,-2)).transpose(-1,-2))
        output = self.ff2(output.transpose(-1,-2)).transpose(-1,-2)
        output = output + residual

        return output



class PositionEncoder(torch.nn.Module):
    def __init__(self, hidden_size, device, drop_prob, max_seq_len=1000):
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

        self.drop_prob = drop_prob


        freq_indices = torch.arange(hidden_size//2)#.repeat_interleave(2)
        frequencies = torch.pow(10000, (2*freq_indices)/hidden_size)

        positions = torch.arange(max_seq_len).reshape(-1, 1)
        positions = positions.repeat(1, hidden_size // 2) # shape (max_seq_len, hidden-size // 2). These are the t values in p_t

        self.position_encodings = torch.zeros((max_seq_len, hidden_size)).to(device)

        self.position_encodings[:, 0::2] = torch.sin(positions / frequencies)
        self.position_encodings[:, 1::2] = torch.cos(positions / frequencies)

    def forward(self, x):
        """

        :param x: tensor with shape (batch_size, seq_len, hidden_size)
        :return:
        """
        # note that we only get the first seq_len position encodings (since max_seq_len
        # may be greater than seq_len)
        output = self.position_encodings[:x.shape[1]] + x
        return F.dropout(output, self.drop_prob, self.training) 



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
        self.char_embed = nn.Embedding.from_pretrained(character_vectors, freeze=False)
        self.word_embed = nn.Embedding.from_pretrained(word_vectors)
        #self.conv1d = nn.Conv1d(in_channels=character_vectors.shape[-1], out_channels=hidden_size // 2, kernel_size=3, padding=1) # not sure on kernel size
        self.conv = nn.Conv2d(in_channels=character_vectors.shape[-1], out_channels = hidden_size, kernel_size=(1,5))
        self.conv_projection = nn.Conv1d(in_channels=word_vectors.shape[-1] + hidden_size, out_channels=hidden_size, kernel_size=1, padding=0, bias=False)
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

        char_emb = char_emb.permute(0, 3, 1, 2) # shape (batch_size, char_embed_size, seq_len, word_len)
        char_emb = self.conv(char_emb) # shape (batch_size, char_embed_size, seq_len, word_len)
        char_emb = F.relu(char_emb) # shape (batch_size, hidden_size, seq_len, word_len) 
        char_emb = torch.max(char_emb, dim=-1)[0] # (batch_size, hidden_size, seq_len)
        char_emb = char_emb.permute(0,2,1) # shape (batch_size, seq_len, hidden_size)
        char_emb = F.dropout(char_emb, self.drop_prob // 2, self.training)

        word_emb = F.dropout(word_emb, self.drop_prob, self.training) # shape (batch_size, seq_len, word_embed_size)

        emb = torch.cat((word_emb, char_emb), dim=2) # (batch_size, seq_len, word_emb_size hidden_size)
        emb = self.conv_projection(emb.transpose(-1,-2)).transpose(-1,-2) # (batch_size, seq_len, hidden_size)

        emb = self.hwy(emb)   # (batch_size, seq_len, hidden_size)
  
        # emb = self.test(emb)

        # Str8AStudent

        return emb
