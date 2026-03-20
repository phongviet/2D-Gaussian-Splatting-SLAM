#!/usr/bin/env bash
set -euo pipefail

python -m pytest -q tests/test_renderer_selection.py tests/test_regularization_gating.py tests/test_representation_shapes.py tests/test_gradient_flow_mock.py
