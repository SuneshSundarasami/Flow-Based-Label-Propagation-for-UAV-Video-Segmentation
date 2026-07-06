# Visualization Scripts

These scripts create local H.264/AAC MP4 comparison videos. The videos are
ignored by git because they are generated artifacts; scripts and small Markdown
summaries are tracked.

All render a 2x2 grid using the selected annotated frames:

```bash
# Original / SEA-RAFT direct / SegProp i01 / ground truth
conda run --no-capture-output -n uav-flowprop python \
    visualizations/create_method_comparison_video.py --video DJI_0101

# Original / forward-only SegProp / standard SegProp i01 / ground truth
conda run --no-capture-output -n uav-flowprop python \
    visualizations/create_segprop_forward_only_video.py --video DJI_0101

# Cache each direct backend separately. These are the only commands that load
# SEA-RAFT or FlowNet2.
PYTHONPATH=src conda run --no-capture-output -n uav-flowprop python -m \
    propagation.backend_runner --video DJI_0101 --backend sea_raft --device cuda
PYTHONPATH=src conda run --no-capture-output -n uav-flowprop python -m \
    propagation.backend_runner --video DJI_0101 --backend flownet2 --device cuda

# Original / cached SEA-RAFT direct / cached FlowNet2 direct / ground truth.
# This command only reads saved masks and metrics; it does not load a flow model.
PYTHONPATH=src conda run --no-capture-output -n uav-flowprop python \
    visualizations/create_flow_backend_comparison_video.py --video DJI_0101
```

The backend runs save prediction caches below
`outputs/backend_predictions/<video>/{sea_raft,flownet2}/`. The final renderer
uses those caches and writes both requested artifacts in one command:

```bash
PYTHONPATH=src conda run --no-capture-output -n uav-flowprop python \
    visualizations/create_flow_backend_comparison_video.py --video DJI_0101
```

Its default panel size is 1920x1080, so the 2x2 MP4 and GIF outputs are
3840x2160 (4K):

```text
visualizations/<video>/searaft_vs_flownet2_direct.mp4
visualizations/<video>/searaft_vs_flownet2_direct_4k.gif
```
