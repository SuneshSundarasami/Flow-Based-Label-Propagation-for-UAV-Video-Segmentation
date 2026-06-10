"""Phase A logic tests (run under the conda env: `pytest`).

These cover the dependency-light pieces that don't need a GPU or the dataset:
config deep-merge, flow colorization shape, and color->index palette mapping.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from config import load_config
from viz import flow_to_color
from data.ruralscapes import (
    Palette, RuralscapesVideo, color_mask_to_index, load_palette,
)
from flow.sea_raft import _load_model_args


def test_config_loads_and_merges():
    cfg = load_config(overrides={"occlusion": {"fb_threshold": 3.0}})
    assert cfg["occlusion"]["fb_threshold"] == 3.0
    # untouched keys keep defaults
    assert "window" in cfg["propagation"]


def test_flow_to_color_shape_and_dtype():
    flow = np.zeros((16, 24, 2), dtype=np.float32)
    flow[..., 0] = 5.0
    img = flow_to_color(flow)
    assert img.shape == (16, 24, 3)
    assert img.dtype == np.uint8


def test_sea_raft_config_parser_loads_with_project_config_imported():
    # Regression check: the project also has a `config` package, so SEA-RAFT's
    # config/parser.py must be loaded by path instead of `import config.parser`.
    cfg_path = (
        Path(__file__).resolve().parents[1]
        / "third_party"
        / "SEA-RAFT"
        / "config"
        / "eval"
        / "spring-M.json"
    )
    args = _load_model_args(cfg_path, iters=2)
    assert args.name == "spring-M"
    assert args.iters == 2


def test_palette_loads():
    pal = load_palette()
    assert pal.num_classes >= 1
    assert pal.ignore_index == 255


def test_color_mask_round_trip():
    pal = Palette(
        ignore_index=255,
        ids=[0, 1],
        names={0: "a", 1: "b"},
        rgb={0: (255, 0, 0), 1: (0, 255, 0)},
    )
    rgb = np.zeros((4, 4, 3), dtype=np.uint8)
    rgb[:2] = (255, 0, 0)   # class 0
    rgb[2:] = (0, 255, 0)   # class 1
    idx = color_mask_to_index(rgb, pal)
    assert (idx[:2] == 0).all()
    assert (idx[2:] == 1).all()


def test_color_mask_unknown_color_is_ignored():
    pal = Palette(255, [0], {0: "a"}, {0: (255, 0, 0)})
    rgb = np.full((2, 2, 3), 7, dtype=np.uint8)  # matches nothing exactly
    idx = color_mask_to_index(rgb, pal, tol=0)
    assert (idx == 255).all()


def test_ruralscapes_fixture_loads_frame_mask_and_pair():
    root = Path(__file__).resolve().parent / "fixtures" / "ruralscapes_demo"
    video = RuralscapesVideo(
        root=root,
        frame_glob="frames/*.ppm",
        mask_glob="masks/*.ppm",
        mask_format="color",
        palette_path=root / "palette.yaml",
    )
    assert len(video) == 2
    assert video.annotated_indices == [1]
    frame = video.load_frame(1)
    mask = video.load_mask(1)
    assert frame.shape == (4, 4, 3)
    assert mask.shape == (4, 4)
    assert set(np.unique(mask).tolist()) == {0, 1, 2}
    pair = video.get_pair(1, 2)
    assert pair.distance == 1
    assert pair.target_mask is None
