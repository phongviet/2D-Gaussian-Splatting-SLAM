from utils.renderer_utils import supports_2d_regularization


def test_regularization_enabled_for_2dgs_only():
    assert supports_2d_regularization("2dgs") is True
    assert supports_2d_regularization("3dgs") is False
