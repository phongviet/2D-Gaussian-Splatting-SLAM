from utils.renderer_utils import resolve_renderer_mode


def test_default_renderer_mode_is_2dgs():
    cfg = {"Training": {}}
    mode = resolve_renderer_mode(cfg, None)
    assert mode == "2dgs"
    assert cfg["Training"]["renderer"] == "2dgs"


def test_renderer_mode_override_to_3dgs():
    cfg = {"Training": {}}
    mode = resolve_renderer_mode(cfg, "3dgs")
    assert mode == "3dgs"
