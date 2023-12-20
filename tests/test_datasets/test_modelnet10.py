
import pytest

from src.datasets import ModelNet10Dataset


@pytest.fixture(scope="session")
def data_root(tmp_path_factory):
    return tmp_path_factory.mktemp("data")


@pytest.fixture(scope="session")
def modelnet10(data_root):
    return ModelNet10Dataset(root=data_root / "modelnet10")


@pytest.mark.slow
def test_length(modelnet10):
    assert len(modelnet10) == 4899


@pytest.mark.slow
def test_shape(modelnet10):
    assert modelnet10[0].shape == (2048, 3)
