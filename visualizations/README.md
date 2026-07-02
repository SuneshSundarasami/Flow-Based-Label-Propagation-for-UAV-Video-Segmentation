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

# Original / SEA-RAFT direct / FlowNet2 direct / ground truth
conda run --no-capture-output -n uav-flowprop python \
    visualizations/create_flow_backend_comparison_video.py --video DJI_0101
```

The last script loads one optical-flow model at a time so it fits on the 6 GB
RTX 3060. Outputs are written under `visualizations/<video>/`; append
`_vscode.mp4` after H.264/AAC conversion when a VS Code preview-friendly copy
is needed.
