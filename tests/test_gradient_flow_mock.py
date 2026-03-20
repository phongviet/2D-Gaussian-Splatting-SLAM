import torch


def test_mock_gradient_flow_parameters_receive_grad():
    xyz = torch.nn.Parameter(torch.randn(10, 3))
    scale = torch.nn.Parameter(torch.randn(10, 2))
    opacity = torch.nn.Parameter(torch.randn(10, 1))
    rot = torch.nn.Parameter(torch.randn(10, 4))

    loss = (xyz**2).mean() + (scale**2).mean() + (opacity**2).mean() + (rot**2).mean()
    loss.backward()

    assert xyz.grad is not None
    assert scale.grad is not None
    assert opacity.grad is not None
    assert rot.grad is not None
