import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.flownet2_pair import _pad_to_multiple_of_64, _pair_tensor


def test_pair_tensor_stacks_rgb_images_for_flownet2():
    img1 = np.zeros((5, 7, 3), dtype=np.uint8)
    img2 = np.ones((5, 7, 3), dtype=np.uint8) * 9

    pair = _pair_tensor(img1, img2, device="cpu")

    assert pair.shape == (1, 3, 2, 5, 7)
    assert pair.dtype == torch.float32
    assert pair[0, 0, 0, 0, 0].item() == 0.0
    assert pair[0, 0, 1, 0, 0].item() == 9.0


def test_pad_to_multiple_of_64_preserves_original_top_left_region():
    pair = torch.arange(1 * 3 * 2 * 65 * 66, dtype=torch.float32).reshape(1, 3, 2, 65, 66)

    padded, shape = _pad_to_multiple_of_64(pair)

    assert shape == (65, 66)
    assert padded.shape == (1, 3, 2, 128, 128)
    assert torch.equal(padded[..., :65, :66], pair)
    assert torch.count_nonzero(padded[..., 65:, :]).item() == 0
