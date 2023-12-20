
from .fixtures import *

import torch


def test_mean(sample_input):
    from src.models.pools import Mean

    mean = Mean()

    sample_output = mean(sample_input)

    expected = list(sample_input.shape)
    expected[1] = 1
    expected = torch.Size(expected)

    actual = sample_output.shape

    assert expected == actual


def test_sum(sample_input):
    from src.models.pools import Sum

    sum = Sum()

    sample_output = sum(sample_input)

    expected = list(sample_input.shape)
    expected[1] = 1
    expected = torch.Size(expected)

    actual = sample_output.shape

    assert expected == actual


@pytest.mark.parametrize("k", [1, 2, 4])
def test_topk(sample_input, k):
    from src.models.pools import TopK

    topk = TopK(k=k)

    sample_output = topk(sample_input)

    expected = list(sample_input.shape)
    expected[1] = k
    expected = torch.Size(expected)

    actual = sample_output.shape

    assert expected == actual


@pytest.mark.parametrize("heads", [2, 4])
@pytest.mark.parametrize("seeds", [2, 32])
@pytest.mark.parametrize("hidden_features", [4, 16])
@pytest.mark.parametrize("output_features", [4, 16])
def test_attention(sample_input, heads, seeds, hidden_features, output_features):
    from src.models.pools import Attention

    input_features = sample_input.shape[-1]
    hidden_features = input_features
    output_features = input_features

    if input_features % heads != 0:
        pytest.skip()

    attention = Attention(
        input_features=input_features,
        hidden_features=hidden_features,
        output_features=output_features,
        heads=heads,
        seeds=seeds
    )

    sample_output = attention(sample_input)

    expected = list(sample_input.shape)
    expected[1] = seeds
    expected[2] = output_features
    expected = torch.Size(expected)

    actual = sample_output.shape

    assert expected == actual

