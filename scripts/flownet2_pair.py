"""Run FlowNet2 on one RGB image pair and write an (H, W, 2) .npy flow array."""
from __future__ import annotations

import argparse
import ctypes
import sys
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import torch

_REPO_ROOT = Path(__file__).resolve().parents[1]
_FLOWNET2_ROOT = _REPO_ROOT / "third_party" / "flownet2-pytorch"

def _load_torch_shared_libs() -> None:
    torch_lib = Path(torch.__file__).resolve().parent / "lib"
    for name in [
        "libc10.so",
        "libtorch.so",
        "libtorch_cpu.so",
        "libtorch_python.so",
        "libc10_cuda.so",
        "libtorch_cuda.so",
    ]:
        path = torch_lib / name
        if path.exists():
            ctypes.CDLL(str(path), mode=ctypes.RTLD_GLOBAL)


def _read_rgb(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"could not read image: {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def _load_model(checkpoint: Path, device: str):
    sys.path.insert(0, str(_FLOWNET2_ROOT))
    _load_torch_shared_libs()
    from models import FlowNet2  # noqa: WPS433

    args = SimpleNamespace(fp16=False, rgb_max=255.0)
    model = FlowNet2(args).to(device)
    state = torch.load(checkpoint, map_location=device)
    model.load_state_dict(state["state_dict"])
    model.eval()
    return model


def _pair_tensor(img1: np.ndarray, img2: np.ndarray, device: str) -> torch.Tensor:
    if img1.shape != img2.shape:
        raise ValueError(f"image shapes differ: {img1.shape} vs {img2.shape}")
    images = np.stack([img1, img2], axis=0).transpose(3, 0, 1, 2)
    return torch.from_numpy(images.astype(np.float32)).unsqueeze(0).to(device)


def _pad_to_multiple_of_64(pair: torch.Tensor) -> tuple[torch.Tensor, tuple[int, int]]:
    _, _, _, height, width = pair.shape
    padded_h = ((height + 63) // 64) * 64
    padded_w = ((width + 63) // 64) * 64
    if padded_h == height and padded_w == width:
        return pair, (height, width)

    padded = torch.zeros(
        (pair.shape[0], pair.shape[1], pair.shape[2], padded_h, padded_w),
        dtype=pair.dtype,
        device=pair.device,
    )
    padded[..., :height, :width] = pair
    return padded, (height, width)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run FlowNet2 on one image pair.")
    parser.add_argument("--img1", required=True)
    parser.add_argument("--img2", required=True)
    parser.add_argument("--out", required=True, help="Output .npy path.")
    parser.add_argument(
        "--checkpoint",
        default="third_party/flownet2-pytorch/checkpoints/FlowNet2_checkpoint.pth.tar",
    )
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available for FlowNet2 inference")

    checkpoint = Path(args.checkpoint)
    if not checkpoint.exists():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint}")

    img1 = _read_rgb(Path(args.img1))
    img2 = _read_rgb(Path(args.img2))
    model = _load_model(checkpoint, args.device)
    pair = _pair_tensor(img1, img2, args.device)
    pair, original_shape = _pad_to_multiple_of_64(pair)
    with torch.no_grad():
        flow = model(pair).squeeze(0)
    flow = flow.detach().cpu().numpy().transpose(1, 2, 0).astype(np.float32)
    flow = flow[: original_shape[0], : original_shape[1]]

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_path, flow)
    print(f"[flownet2-pair] wrote {out_path} shape={flow.shape}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
