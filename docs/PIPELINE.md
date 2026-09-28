# Full command pipeline

All examples assume an activated environment with `python -m pip install -e '.[notebooks]'` and these paths:

```bash
SPLIT_ROOT=/data/JRDB/test
SPLIT_NAME=test
ODOMETRY_ROOT=/data/JRDB/odometry
OUTPUT_ROOT=/data/tracker_outputs
```

## Yaw conventions for local and global runs

Both GT and the original PersonMinkUNet detection files default to
`jrdb_clockwise` as JRDB yaw is CW encoded. Local-only runs decode source yaw as `yaw_local_ccw = -rot_z`
without changing centers or requiring odometry. Global runs use
`yaw_global_ccw = -rot_z + ego_yaw` and transform centers with the pose.

For an external detector whose saved output is already CCW, add
`--detection-yaw-convention standard_ccw --gt-yaw-convention jrdb_clockwise` to the
tracker command. Its yaw is then `rot_z` locally or `rot_z + ego_yaw` globally.
Both flags work with or without `--global_coords`; decoding happens exactly once.

## 1. Optional measured score distribution

The packaged pseudo specification uses constant confidence `1.0`, so it runs without another artifact. To sample the same measured score realization across conditions, first build a distribution and change `score_mode` to `sample` or pass the resulting file with `--score_dists`:

```bash
python -m tracker_eval.cli.build_score_distributions_from_gt_det \
  --help
```

## 2. Pseudo detections

```bash
tracker-eval-generate-pseudo \
  --split_root "$SPLIT_ROOT" \
  --split_name "$SPLIT_NAME" \
  --labels_subdir labels_3d \
  --odometry_root "$ODOMETRY_ROOT" \
  --spec tracker_eval/cli/pseudo_det_spec.yaml \
  --out_detections_subdir detections_3D_pseudo
```

The generator is deterministic for a fixed seed/spec/GT and incrementally reuses compatible JSON files. It writes one subdirectory per clean or corrupted condition plus `manifest.json` and `generation_status.csv`. Every condition is evaluated against unchanged standard GT.

## 3. Sequential tracker execution

```bash
tracker-eval \
  --split_root "$SPLIT_ROOT" \
  --split_name "$SPLIT_NAME" \
  --out_root "$OUTPUT_ROOT" \
  --trackers pedreftrack \
  --pedreftrack_modes no_gt gt_assisted \
  --global_coords --odometry_root "$ODOMETRY_ROOT"
```

Use sequential mode for FPS, mean latency and latency quantiles. `frame_stats/*.csv` measures only the tracker step; loading/export are outside that interval.

## 4. Shared parallel scheduler

```bash
tracker-eval \
  --split_root "$SPLIT_ROOT" \
  --split_name "$SPLIT_NAME" \
  --out_root "$OUTPUT_ROOT" \
  --trackers cbmot elptnet fastpoly gnnpmb simpletrack pedreftrack \
  --parallel --num_workers 12 \
  --global_coords --odometry_root "$ODOMETRY_ROOT"
```

The pool schedules sequence jobs across all requested trackers. Each sequence gets a fresh tracker instance. This maximizes throughput but deliberately reports no meaningful tracker timing or per-frame profile. Do not use its apparent wall-clock rate in runtime comparisons.

## 5. Manifest-driven tracker runs

```bash
tracker-eval-pseudo \
  --manifest "$SPLIT_ROOT/detections_3D_pseudo/manifest.json" \
  --out_root "$OUTPUT_ROOT" \
  --trackers cbmot elptnet fastpoly gnnpmb simpletrack pedreftrack \
  --pedreftrack_modes no_gt \
  --parallel --num_workers 12 \
  --global_coords
```

Output names follow `<tracker>__global[_pedreftrack_mode]_<variant>` throughout running, evaluation and plotting; no alternate ordering or iteration tag is used.

The manifest supplies the detection subdirectory for every condition and the unchanged labels directory. PedRefTrack `gt_assisted` may be selected explicitly for diagnostic/reference runs; no other adapter receives GT.

## 6. Evaluation GT

Local and global TrackEval GT are separate exported representations of the same label JSON:

```bash
tracker-eval-prepare-gt \
  --labels-dir "$SPLIT_ROOT/labels_3d" \
  --gt-folder /data/eval_gt_local \
  --split "$SPLIT_NAME"

tracker-eval-prepare-gt \
  --labels-dir "$SPLIT_ROOT/labels_3d" \
  --gt-folder /data/eval_gt_global \
  --split "$SPLIT_NAME" \
  --global-coords --odometry-root "$ODOMETRY_ROOT"
```

Use local GT for local tracker outputs and global GT for names containing `__global`.

## 7. Standard metrics

```bash
tracker-eval-evaluate \
  --gt-folder /data/eval_gt_global \
  --trackers-folder "$OUTPUT_ROOT" \
  --output-folder /data/evaluation_results \
  --split "$SPLIT_NAME" \
  --trackers cbmot__global fastpoly__global pedreftrack__global_no_gt
```

The evaluator uses HOTA, CLEAR at 3D similarity threshold `0.3`, Identity and Count. Its default tracker subfolder is `<split>/data`, matching runner output. Override `--tracker-subfolder` only for a reorganized TrackEval-style tree.

## 8. Pseudo metrics and capability tables

```bash
tracker-eval-protocol pseudo \
  --trackers-dir "$OUTPUT_ROOT" \
  --gt-folder /data/eval_gt_global \
  --output-dir /data/pseudo_results \
  --split "$SPLIT_NAME" --workers 12
```

```bash
tracker-eval-protocol capabilities \
  --trackers-base-dir "$OUTPUT_ROOT" \
  --gt-folder /data/eval_gt_global \
  --local-gt-folder /data/eval_gt_local \
  --detections-dir "$SPLIT_ROOT/detections_3D" \
  --output-dir /data/protocol_results \
  --trackers pedreftrack__global_gt_assisted,pedreftrack__global_no_gt,cbmot__global,fastpoly__global,gnnpmb__global,simpletrack__global,elptnet__global \
  --reference-tracker pedreftrack__global_gt_assisted \
  --split "$SPLIT_NAME" --workers 8
```

Add `--recompute-trackers name1,name2` to refresh selected caches or `--force` after changing protocol-defining settings.

The capability command uses the corrected RAL defaults: a `2.0` s gap horizon and recovery-bin edges `1,4,7,10,13,16,19,22,25,28,31`. These and the other scientific protocol settings are exposed as optional arguments on `tracker-eval-protocol capabilities`; they do not need to be repeated for the standard evaluation.

The initialization table already provides both requested populations:

- `event_type == "trajectory_start"`: initial trajectory establishment only;
- `event_type == "all"`: initial establishment plus post-gap reacquisition;
- `event_type == "post_gap_reacquisition"`: reacquisition alone, retained for diagnostics.

Set `INITIALIZATION_EVENT_TYPE` in `notebooks/plot_tracker_results.ipynb` to select the plotted population. Both views are calculated in the same protocol run.

## 9. Plots

```bash
jupyter lab notebooks/
```

Configure result/data paths in the first cell, then run all cells in the relevant notebook. The notebooks read cached tables/results; they do not rerun tracker inference.
