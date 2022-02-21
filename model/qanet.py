import torch

from context_query_attention import ContextQueryAttention
from encoder_block import EncoderBlock


class QANet(torch.nn.Module):
    """QANet prediction network, as described in the QANet paper.

    See https://arxiv.org/pdf/1804.09541.pdf for more details.
    """

    def __init__(self):
        super(QANet, self).__init__()
