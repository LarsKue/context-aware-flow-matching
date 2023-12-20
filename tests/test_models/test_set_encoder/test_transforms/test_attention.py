
import pytest

import torch


@pytest.fixture
def sample_input():
    return torch.rand(4, 2048, 3)


@pytest.mark.skip("Currently broken")
def test_mab(sample_input):
    from src.models.transforms.attention import MultiheadAttentionBlock

    mab = MultiheadAttentionBlock(64, 64, 64, heads=8)

    sample_output = mab(sample_input, sample_input)

    assert sample_output.shape == (4, 2048, 64)


