import torch


class ContextQueryAttention(torch.nn.Module):
    """Context-query attention subnetwork, as described in the QANet paper.

    See https://arxiv.org/pdf/1804.09541.pdf for more details.
    """

    def __init__(self):
        super(ContextQueryAttention, self).__init__()
