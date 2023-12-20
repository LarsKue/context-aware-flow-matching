
import pytest

import torch


@pytest.fixture(scope="session", params=[4, 16])
def batch_size(request):
    return request.param


@pytest.fixture(scope="session", params=[16, 256])
def set_size(request):
    return request.param


@pytest.fixture(scope="session", params=[4, 16])
def features(request):
    return request.param


@pytest.fixture(scope="session")
def sample_input(batch_size, set_size, features):
    return torch.rand(batch_size, set_size, features)
