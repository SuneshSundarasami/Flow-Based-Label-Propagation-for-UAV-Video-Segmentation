import csv
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from data.ruralscapes import load_palette
from scripts.prepare_segprop_dataset import (
    _build_rgb_lut,
    _color_mask_to_index_fast,
    _one_hot,
    _segprop_label_name,
    prepare_labels,
)


def test_segprop_label_name_uses_video_and_zero_padded_frame():
    path = Path("segfull_DJI_0043_50.png")
    assert _segprop_label_name(path) == "DJI_0043_000050.npz"


def test_one_hot_preserves_known_classes_and_leaves_ignore_empty():
    mask = np.array([[0, 1], [255, 1]], dtype=np.int32)
    encoded = _one_hot(mask, num_classes=3)

    assert encoded.shape == (2, 2, 3)
    assert encoded.dtype == bool
    assert encoded[0, 0, 0]
    assert encoded[0, 1, 1]
    assert not encoded[1, 0].any()


def test_fast_color_lookup_maps_palette_and_unknown_to_ignore():
    palette = load_palette()
    rgb = np.array(
        [
            [palette.rgb[0], palette.rgb[1]],
            [(3, 4, 5), palette.rgb[0]],
        ],
        dtype=np.uint8,
    )
    mapped = _color_mask_to_index_fast(rgb, _build_rgb_lut(palette))

    assert mapped.tolist() == [[0, 1], [palette.ignore_index, 0]]


def test_prepare_labels_writes_all_even_odd_and_metadata(tmp_path):
    dataset_root = tmp_path / "Ruralscapes"
    label_dir = dataset_root / "labels" / "manual_labels" / "DJI_TEST"
    label_dir.mkdir(parents=True)

    palette = load_palette()
    class0 = palette.rgb[0]
    class1 = palette.rgb[1]
    for frame_index, top_color in [(0, class0), (50, class1)]:
        rgb = np.zeros((2, 3, 3), dtype=np.uint8)
        rgb[0, :] = top_color
        rgb[1, :] = class1
        Image.fromarray(rgb, mode="RGB").save(label_dir / f"segfull_DJI_TEST_{frame_index}.png")

    out_root = tmp_path / "segprop"
    counts = prepare_labels(
        dataset_root=dataset_root,
        out_root=out_root,
        videos=["DJI_TEST"],
        size=(4, 2),
        compress=False,
        overwrite=False,
    )

    assert counts == {"DJI_TEST": 2}
    first = out_root / "labels_2k" / "all" / "DJI_TEST" / "DJI_TEST_000000.npz"
    second = out_root / "labels_2k" / "all" / "DJI_TEST" / "DJI_TEST_000050.npz"
    assert first.exists()
    assert second.exists()
    assert (out_root / "labels_2k" / "train_even" / "DJI_TEST" / first.name).exists()
    assert (out_root / "labels_2k" / "train_odd" / "DJI_TEST" / second.name).exists()

    with np.load(first) as data:
        assert set(data.files) == {"map", "votes"}
        assert data["map"].shape == (2, 4, palette.num_classes)
        assert data["map"].dtype == bool
        assert np.array_equal(data["map"], data["votes"])

    metadata_path = out_root / "metadata" / "DJI_TEST_labels.csv"
    with open(metadata_path, newline="") as fh:
        rows = list(csv.DictReader(fh))
    assert [row["frame_index"] for row in rows] == ["0", "50"]
    assert [row["split"] for row in rows] == ["train_even", "train_odd"]
    assert rows[0]["segprop_label"] == "labels_2k/all/DJI_TEST/DJI_TEST_000000.npz"
