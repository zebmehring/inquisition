import torch


class PositionEncoder(torch.nn.Module):
    def __init__(self):
        super(PositionEncoder, self).__init__()


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

    def __init__(self):
        super(EncoderBlock, self).__init__()
