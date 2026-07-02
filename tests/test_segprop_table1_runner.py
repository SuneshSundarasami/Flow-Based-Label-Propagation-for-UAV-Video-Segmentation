import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.run_segprop_table1 import (
    evaluate_table1,
    frame_copy,
    frame_filter,
    paths_for_video,
    run_video_table1,
    validate_inputs,
    write_table,
)


class FakeSegProp:
    def __init__(self):
        self.calls = []

    def vote(self, flow_h5, gt_path, output_path, **kwargs):
        self.calls.append(("vote", flow_h5, gt_path, output_path, kwargs))
        Path(output_path).mkdir(parents=True, exist_ok=True)

    def iterate(self, flow_h5, pv_path, output_path, **kwargs):
        self.calls.append(("iterate", flow_h5, pv_path, output_path, kwargs))
        Path(output_path).mkdir(parents=True, exist_ok=True)

    def denoise(self, flow_h5, pv_path, output_path, **kwargs):
        self.calls.append(("denoise", flow_h5, pv_path, output_path, kwargs))
        Path(output_path).mkdir(parents=True, exist_ok=True)


class FakeStats:
    def __init__(self):
        self.calls = []

    def evaluate(self, test_path, label_path):
        self.calls.append((test_path, label_path))
        if test_path.endswith("filtered"):
            return 0.903, 0.829
        return 0.884, 0.801


def _touch_inputs(out_root: Path, video: str = "DJI_TEST") -> None:
    for split in ["train_even", "train_odd"]:
        label_dir = out_root / "labels_2k" / split / video
        label_dir.mkdir(parents=True, exist_ok=True)
        np.savez(label_dir / f"{video}_000000.npz", map=np.zeros((2, 2, 1), dtype=bool))
    flow_dir = out_root / "flow_2k_fn2"
    flow_dir.mkdir(parents=True, exist_ok=True)
    (flow_dir / f"{video}_forward.h5").write_bytes(b"")
    (flow_dir / f"{video}_backward.h5").write_bytes(b"")


def test_frame_copy_and_filter_match_paper_demo():
    assert frame_copy(0)
    assert frame_copy(100)
    assert not frame_copy(50)
    assert not frame_filter(50)
    assert frame_filter(0)
    assert frame_filter(51)
    assert frame_filter(100)


def test_validate_inputs_requires_labels_and_flows(tmp_path):
    paths = paths_for_video(tmp_path, "DJI_TEST")
    try:
        validate_inputs(paths)
    except FileNotFoundError as exc:
        assert "missing SegProp Table 1 inputs" in str(exc)
    else:
        raise AssertionError("validate_inputs should fail for missing inputs")

    _touch_inputs(tmp_path)
    validate_inputs(paths_for_video(tmp_path, "DJI_TEST"))


def test_run_video_table1_calls_vote_iterations_and_denoise(tmp_path):
    _touch_inputs(tmp_path)
    fake = FakeSegProp()

    filtered = run_video_table1(
        video="DJI_TEST",
        paths=paths_for_video(tmp_path, "DJI_TEST"),
        segprop_module=fake,
        iterations=3,
        device="cpu",
        overwrite=False,
        prune_intermediate=False,
    )

    assert filtered == tmp_path / "output_2k" / "filtered" / "DJI_TEST"
    assert [call[0] for call in fake.calls] == ["vote", "iterate", "iterate", "denoise"]
    assert fake.calls[0][4]["precalc_flow"] is False
    assert fake.calls[1][4]["pv_series"] == [0, 5, 10]
    assert fake.calls[-1][4]["pv_series"] == [0, 1, 3, 5, 7]
    assert fake.calls[-1][4]["method"] == "self"


def test_evaluate_and_write_table(tmp_path):
    _touch_inputs(tmp_path)
    (tmp_path / "output_2k" / "i01" / "DJI_TEST").mkdir(parents=True)
    (tmp_path / "output_2k" / "filtered" / "DJI_TEST").mkdir(parents=True)

    fake_stats = FakeStats()
    rows = evaluate_table1(out_root=tmp_path, stats_module=fake_stats, max_iteration=1)
    assert [row["stage"] for row in rows] == ["i01", "filtered"]
    assert rows[0]["paper_mf1"] == "0.884000"
    assert rows[1]["delta_miou"] == "0.000000"
    assert all(Path(test_path).is_absolute() for test_path, _ in fake_stats.calls)
    assert all(Path(label_path).is_absolute() for _, label_path in fake_stats.calls)

    csv_path, md_path = write_table(rows, tmp_path)
    with open(csv_path, newline="") as fh:
        parsed = list(csv.DictReader(fh))
    assert parsed[0]["stage"] == "i01"
    assert "filtered" in md_path.read_text()
