import torch

from torch.nn.functional import softmax


class ContextQueryAttention(torch.nn.Module):
    """Context-query attention subnetwork, as described in the QANet paper.

    See https://arxiv.org/pdf/1804.09541.pdf for more details.
    """

    def __init__(self, embedding_dim):
        super(ContextQueryAttention, self).__init__()
        self.W = torch.nn.Parameter(torch.empty(embedding_dim))

    def similarity(self, context, query):
        return torch.cat((context, query, torch.einsum('bnd,bmd->bnmd', context, query)), dim=-1) @ self.W

    def forward(self, context, query):
        similarity = self.similarity(context, query)
        similarity = softmax(similarity, dim=1)
        A = similarity @ query.T
        B = similarity @ softmax(similarity, dim=0).T @ context.T
        return A, B
