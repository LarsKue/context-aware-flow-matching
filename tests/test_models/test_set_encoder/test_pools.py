
import pytest

import torch


@pytest.fixture(scope="session")
def sample_input():
    return torch.rand(4, 2048, 3)


