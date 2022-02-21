import torch

from position_encoder import PositionEncoder

from torch.nn import Conv2d
from torch.nn import ModuleList
from torch.nn.functional import layer_norm
from torch.nn.functional import relu
from torch.nn.functional import softmax


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

        self.hidden_size = hidden_size

        self.position_encoder = PositionEncoder()

        self.convs = ModuleList([Conv2d(in_channels=hidden_size, out_channels=hidden_size, kernel_size=7)
                                 for _ in range(num_convs)])

        # TODO: Support multi-headed attention
        self.Q = torch.nn.Linear(
            in_features=hidden_size, out_features=hidden_size)
        self.K = torch.nn.Linear(
            in_features=hidden_size, out_features=hidden_size)
        self.V = torch.nn.Linear(
            in_features=hidden_size, out_features=hidden_size)

        self.ff = torch.nn.Linear(
            in_features=hidden_size, out_features=hidden_size)

    def forward(self, x):
        output = self.position_encoder(x)

        for conv in self.convs:
            residual = output
            output = layer_norm(output, normalized_shape=self.hidden_size)
            output = conv(output)
            output += residual

        residual = output
        output = layer_norm(output, normalized_shape=self.hidden_size)
        output = softmax(self.Q(output) @ self.K(output).T /
                         torch.sqrt(self.hidden_size)) @ self.V(output)
        output += residual

        residual = output
        output = layer_norm(output, normalized_shape=self.hidden_size)
        output = relu(self.ff(output))
        output += residual

        return output
