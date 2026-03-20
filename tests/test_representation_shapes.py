import torch


def test_2d_surfel_scaling_shape():
    scaling = torch.randn(8, 2)
    assert scaling.shape[1] == 2


def test_3d_gaussian_scaling_shape():
    scaling = torch.randn(8, 3)
    assert scaling.shape[1] == 3
