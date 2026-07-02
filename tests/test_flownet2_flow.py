import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from flow.flownet2 import FlowNet2Flow


def test_flownet2_padding_preserves_original_shape():
    pair = torch.zeros((1, 3, 2, 1080, 2048))
    padded, original = FlowNet2Flow._pad_to_multiple_of_64(pair)
    assert original == (1080, 2048)
    assert padded.shape == (1, 3, 2, 1088, 2048)


def test_flownet2_padding_skips_aligned_shape():
    pair = torch.zeros((1, 3, 2, 1024, 2048))
    padded, original = FlowNet2Flow._pad_to_multiple_of_64(pair)
    assert original == (1024, 2048)
    assert padded.shape == pair.shape
