import sys
from pathlib import Path

import cv2
import h5py
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.generate_flownet2_h5 import (
    generate_video_flows,
    list_frames,
    validate_flow,
)


def _write_frame(path: Path, value: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = np.full((3, 4, 3), value, dtype=np.uint8)
    assert cv2.imwrite(str(path), image)


def test_list_frames_sorts_by_numeric_index(tmp_path):
    root = tmp_path / "frames" / "DJI_TEST"
    _write_frame(root / "DJI_TEST_000010.jpg", 10)
    _write_frame(root / "DJI_TEST_000002.jpg", 2)
    _write_frame(root / "DJI_TEST_000001.jpg", 1)

    frames = list_frames(root)

    assert [frame.index for frame in frames] == [1, 2, 10]


def test_validate_flow_requires_shape_and_finite_values():
    flow = np.zeros((3, 4, 2), dtype=np.float64)
    assert validate_flow(flow, (3, 4)).dtype == np.float32

    with pytest.raises(ValueError, match="expected flow shape"):
        validate_flow(np.zeros((3, 4), dtype=np.float32), (3, 4))
    bad = np.zeros((3, 4, 2), dtype=np.float32)
    bad[0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        validate_flow(bad, (3, 4))


def test_generate_video_flows_writes_forward_backward_h5_and_progress(tmp_path):
    frames_root = tmp_path / "frames_2k"
    for index, value in enumerate([20, 40, 60]):
        _write_frame(frames_root / "DJI_TEST" / f"DJI_TEST_{index:06d}.jpg", value)

    helper = tmp_path / "fake_flow.py"
    helper.write_text(
        "import cv2, numpy as np, sys\n"
        "img1, img2, out = sys.argv[1:4]\n"
        "a = cv2.imread(img1, cv2.IMREAD_GRAYSCALE)\n"
        "b = cv2.imread(img2, cv2.IMREAD_GRAYSCALE)\n"
        "flow = np.zeros((*a.shape, 2), dtype=np.float32)\n"
        "flow[..., 0] = (float(b.mean()) - float(a.mean())) / 20.0\n"
        "flow[..., 1] = 2.0\n"
        "np.save(out, flow)\n"
    )

    out_root = tmp_path / "out"
    command = f"{sys.executable} {helper} {{img1}} {{img2}} {{out}}"
    forward, backward = generate_video_flows(
        video="DJI_TEST",
        frames_root=frames_root,
        out_root=out_root,
        flow_command=command,
    )

    with h5py.File(forward, "r") as h5:
        data = h5["flow"][:]
        assert data.shape == (2, 3, 4, 2)
        assert np.allclose(data[..., 0], 1.0)
        assert np.allclose(data[..., 1], 2.0)
    with h5py.File(backward, "r") as h5:
        data = h5["flow"][:]
        assert data.shape == (2, 3, 4, 2)
        assert np.allclose(data[..., 0], -1.0)
    assert (out_root / "flow_2k_fn2" / "DJI_TEST_progress.json").exists()
