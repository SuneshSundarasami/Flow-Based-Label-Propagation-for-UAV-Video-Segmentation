# DJI_0101 Comparisons

Generated MP4 files in this directory are ignored. The current scripts render
these 2x2 grids:

- `searaft_vs_segprop_i01_vscode.mp4`: original, direct SEA-RAFT, SegProp i01,
  and ground truth.
- `segprop_forward_only_vs_i01_vscode.mp4`: original, forward-only SegProp,
  standard SegProp i01, and ground truth.
- `searaft_vs_flownet2_direct_vscode.mp4`: original, direct SEA-RAFT, direct
  FlowNet2, and ground truth.

For the matched seven-pair 2K direct-propagation check, SEA-RAFT obtained
all-pixel / valid-pixel mIoU `0.825929 / 0.873571` with `77.971%` valid
coverage. FlowNet2 obtained `0.828829 / 0.874143` with `68.737%` coverage.

SegProp i01 on the full sequence obtained mF1/mIoU `0.899522 / 0.824061`; the
filtered result obtained `0.909171 / 0.841056`.
