import torch
from torch.utils.data import Dataset

from src.data import corrupt_mnist
from src.model import FredNet
from tests import _PATH_DATA, _PROJECT_ROOT, _TEST_ROOT


def test_data():
    train, test = corrupt_mnist()
    assert len(train) == 30000
    assert len(test) == 5000
    for dataset in [train, test]:
        for x, y in dataset:
            assert x.shape == (1, 28, 28)
            assert y in range(10)
    train_targets = torch.unique(train.tensors[1])
    assert (train_targets == torch.arange(0,10)).all()
    test_targets = torch.unique(test.tensors[1])
    assert (test_targets == torch.arange(0,10)).all()

def test_model():
    model = FredNet()
    x = torch.randn(1, 1, 28, 28)
    y = model(x)
    assert y.shape == (1, 10)
