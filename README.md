# 3D Ellipsoid Visualization

To run the 3D ellipsoid visualization script, use the following command:

```bash
python ellipsoid_3d_standalone.py \
  --frame-root ../Campus_Seq1/frames \
  --calibration ../Campus_Seq1/calibration.json \
  --pose-file ../Campus_Seq1/result_3d.json \
  --output-video ./results/ellipsoid_plot.mp4
```

- Adjust the paths as needed for your dataset location.
- The output video will be saved to `./results/ellipsoid_plot.mp4` by default.
- Additional options:
  - `--max-frames N` to limit the number of frames processed.
  - `--cameras Camera0 Camera1 ...` to select specific cameras.