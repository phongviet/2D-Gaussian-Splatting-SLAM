import numpy as np
import torch

from gui.video_recorder import (
    depth_frame_to_uint8,
    depth_range_from_gt,
    normal_frame_to_uint8,
    rgb_frame_to_uint8,
)


def test_rgb_frame_to_uint8_accepts_chw_tensor():
    image = torch.tensor(
        [
            [[0.0, 0.5], [1.0, 2.0]],
            [[1.0, 0.5], [0.0, -1.0]],
            [[0.25, 0.5], [0.75, 1.0]],
        ]
    )

    result = rgb_frame_to_uint8(image)

    assert result.shape == (2, 2, 3)
    assert result.dtype == np.uint8
    assert result[0, 0].tolist() == [0, 255, 63]
    assert result[1, 0].tolist() == [255, 0, 191]


def test_depth_frame_to_uint8_accepts_single_channel_tensor():
    depth = torch.tensor([[[0.0, 0.2], [1.0, 2.0]]])

    result = depth_frame_to_uint8(depth)

    assert result.shape == (2, 2, 3)
    assert result.dtype == np.uint8
    assert result[0, 0].tolist() == [0, 0, 0]


def test_depth_range_from_gt_matches_readme_percentile_scale():
    gt_depth = np.array([[0.0, 1.0, 2.0], [3.0, 4.0, 100.0]], dtype=np.float32)

    min_value, max_value = depth_range_from_gt(gt_depth)

    assert np.isclose(min_value, np.percentile(gt_depth[gt_depth > 0], 1))
    assert np.isclose(max_value, np.percentile(gt_depth[gt_depth > 0], 99))


def test_normal_frame_to_uint8_maps_normals_to_color():
    normal = torch.tensor(
        [
            [[-1.0, 0.0]],
            [[0.0, 1.0]],
            [[1.0, -1.0]],
        ]
    )

    result = normal_frame_to_uint8(normal)

    assert result.shape == (1, 2, 3)
    assert result.dtype == np.uint8
    assert result[0, 0].tolist() == [0, 127, 255]
    assert result[0, 1].tolist() == [127, 255, 0]
