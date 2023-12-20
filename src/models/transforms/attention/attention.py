
import torch
import torch.nn as nn

from torch import Tensor


class Attention(nn.Module):
    def forward(self, query: Tensor, key: Tensor, value: Tensor):
        m = torch.matmul(query, key.transpose(-2, -1))
        m = m * query.size(-1) ** -0.5
        m = torch.softmax(m, dim=-1)
        m = torch.matmul(m, value)

        return m


class MultiheadAttention(nn.Module):
    def __init__(self, query_features: int, key_features: int, value_features: int, heads: int):
        super().__init__()
        self.register_buffer("query_features", torch.tensor(query_features, dtype=torch.int64))
        self.register_buffer("key_features", torch.tensor(key_features, dtype=torch.int64))
        self.register_buffer("value_features", torch.tensor(value_features, dtype=torch.int64))
        self.register_buffer("heads", torch.tensor(heads, dtype=torch.int64))

        self.WQ = nn.Parameter(torch.empty(heads, query_features, key_features))
        nn.init.xavier_uniform_(self.WQ)
        self.WK = nn.Parameter(torch.empty(heads, query_features, key_features))
        nn.init.xavier_uniform_(self.WK)
        self.WV = nn.Parameter(torch.empty(heads, query_features, value_features))
        nn.init.xavier_uniform_(self.WV)
        self.WO = nn.Parameter(torch.empty(heads, value_features, value_features))
        nn.init.xavier_uniform_(self.WO)

        self.att = Attention()

    def forward(self, query: Tensor, key: Tensor, value: Tensor):
        query = torch.einsum("BSQ, HQK -> BHSK", query, self.WQ)
        key = torch.einsum("BSK, HKQ -> BHSQ", key, self.WK)
        value = torch.einsum("BSV, HVQ -> BHSQ", value, self.WV)

        heads = self.att(query, key, value)

        return torch.einsum("BHSV, HVQ -> BSQ", heads, self.WO)


q = torch.randn(2, 3, 4)
k = torch.randn(2, 3, 5)
v = torch.randn(2, 3, 6)
att = MultiheadAttention(4, 5, 6, 7)
y = att(q, k, v)

print(y.shape)
