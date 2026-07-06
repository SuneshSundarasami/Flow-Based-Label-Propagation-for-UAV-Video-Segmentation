# Ruralscapes video splits

These splits are video-level, so frames from one video never appear in more
than one training/validation/test group.

- `ruralscapes_geoseg_train.txt`: 11 videos for learned GeoSeg training.
- `ruralscapes_geoseg_val.txt`: 1 video (`DJI_0050`) for model selection and early stopping.
- `ruralscapes_geoseg_test.txt`: 5 well-annotated held-out videos for the primary result.
- `ruralscapes_geoseg_test_supplementary.txt`: 2 held-out videos with fewer annotations.
- `ruralscapes_development_smoke.txt`: `DJI_0101`, reserved for pipeline development;
  do not include it in training if it is used for debugging or visual comparisons.

The five primary test videos have the strongest annotation counts among the
official Ruralscapes testing split: `DJI_0056`, `DJI_0061`, `DJI_0086`,
`DJI_0089`, and `DJI_0116`.
