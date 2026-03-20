def resolve_renderer_mode(config, cli_renderer=None):
    if cli_renderer:
        mode = cli_renderer.lower()
    else:
        mode = str(config.get("Training", {}).get("renderer", "2dgs")).lower()
    if mode not in {"2dgs", "3dgs"}:
        raise ValueError(f"Unsupported renderer mode: {mode}")
    config.setdefault("Training", {})["renderer"] = mode
    return mode


def get_renderer_components(mode):
    if mode == "3dgs":
        from gaussian_splatting.gaussian_renderer.render_3d import render
        from gaussian_splatting.scene.gaussian_model_3d import GaussianModel

        return render, GaussianModel

    from gaussian_splatting.gaussian_renderer.render_2d import render
    from gaussian_splatting.scene.gaussian_model import GaussianModel

    return render, GaussianModel


def supports_2d_regularization(mode):
    return mode == "2dgs"
