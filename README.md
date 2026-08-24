# tracker_eval

`tracker_eval` defines and implements a deployment-oriented evaluation protocol for 3D pedestrian tracking under fixed, shared detections. The protocol complements HOTA with directly interpretable, tracker-only success profiles that characterize trajectory initialization, continuation through missed detections, post-gap identity recovery, close-neighbour continuity, and tracker-step runtime as a function of input load. The repository provides the complete pipeline for running 3D trackers to generate trajectory outputs on JRDB dataset, evaluating them with the bundled corrected TrackEval runtime, computing the capability protocol using the success profile metrics, and plotting the saved results.

To make the protocol reproducible and usable out of the box, the repository includes compact, evaluation-oriented runtime subsets of seven example trackers: AB3DMOT, CBMOT, ELPTNet, FastPoly, GNN-PMB, SimpleTrack, and PedRefTrack. These implementations serve as ready-to-run reference methods for applying the protocol, they are not intended to replace the trackers’ full official repositories.

The general ROS2 implementation of PedRefTrack is maintained separately at https://github.com/SCAI-Lab/PedRefTrack. This repository includes the same pure-Python tracking core together with a thin evaluation adapter, while the separate PedRefTrack repository provides the deployable ROS2 node operating on `Detection3DArray` messages.

## Installation

Python 3.10 or newer is required.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

For the notebooks, install the optional tools:

```bash
python -m pip install -e '.[notebooks]'
```

GNN-PMB additionally needs its Murty C++ extension. The repository contains the architecture-neutral source and does not ship the binary.

```bash
sudo apt install cmake libeigen3-dev
tracker-eval-build gnnpmb
```

The helper performs an out-of-source CMake build with the active Python interpreter and installed `pybind11`, then places `_murty*.so` beside the GNN-PMB package. The generated binary and build directory are ignored by Git.

## Included trackers

| CLI name | Compact runtime included | Authoritative upstream |
|---|---:|---|
| `ab3dmot` | yes | [xinshuoweng/AB3DMOT](https://github.com/xinshuoweng/AB3DMOT) |
| `cbmot` | yes | [cogsys-tuebingen/CBMOT](https://github.com/cogsys-tuebingen/CBMOT) |
| `elptnet` | yes, box tracker path | [jinzhengguang/ELPTNet](https://github.com/jinzhengguang/ELPTNet) |
| `fastpoly` | yes | [lixiaoyu2000/FastPoly](https://github.com/lixiaoyu2000/FastPoly) |
| `gnnpmb` | yes; build Murty once | [chisyliu/GnnPmbTracker](https://github.com/chisyliu/GnnPmbTracker) |
| `simpletrack` | yes | [tusen-ai/SimpleTrack](https://github.com/tusen-ai/SimpleTrack) |
| `pedreftrack` | yes | [SCAI-Lab/PedRefTrack](https://github.com/SCAI-Lab/PedRefTrack) |

These are compact protocol adapters and required runtime files, not forks intended to replace the upstream projects. Config files used by each adapter are retained in `tracker_eval/trackers/implementations/<tracker>/`.

## Input data layout

Each split is a directory containing one JSON file per sequence:

```text
<split_root>/
├── detections_3D/
│   └── <sequence>.json
└── labels_3d/
    └── <sequence>.json
```

For global-coordinate runs, odometry is read from:

```text
<odometry_root>/<split_name>/odometry/<sequence>.csv
```

The CSV columns are `timestamp_ns,x,y,z,qx,qy,qz,qw`. Row `i` is deliberately matched to frame index `i`; timestamps are used as tracker timestamps but not to look up a pose row.

### Detection and source-GT JSON

The accepted top-level detection containers are `detections`, `dets`, or `predictions`. GT accepts `labels`, `annotations`, `frames`, or `data`. A direct frame-to-list mapping is also accepted. Frame keys may be integer-like strings or names such as `000123.pcd`.

```json
{
  "detections": {
    "000000.pcd": [
      {
        "box": {
          "cx": 2.1, "cy": -0.4, "cz": 0.85,
          "l": 0.5, "w": 0.5, "h": 1.7, "rot_z": 0.0
        },
        "score": 0.91,
        "label_id": "pedestrian:-1"
      }
    ]
  }
}
```

`box` may instead be the list `[cx, cy, cz, l, w, h, rot_z]`. The internal convention is a center-based box in metres/radians with x forward, y left, z up, length along x, width along y and yaw about +z. Detection `score` and `label_id` are optional; detections default to track ID `-1`. GT uses `label_id: "pedestrian:<integer>"` as the persistent identity. An explicit integer `track_id` overrides the ID parsed from `label_id`.

`tracker_eval/data/jrdb_io.py` implements this parsing through `load_jrdb_detections_3d`, `load_jrdb_labels_3d` and `_parse_entry_to_detection`.

Ordinary trackers iterate the frame keys present in the detection JSON. GT is loaded only for a tracker that exposes `step_with_gt`; in this repository that is PedRefTrack. PedRefTrack `no_gt` does not use labels, while `gt_assisted` does. Pseudo-detection JSON retains every source GT frame key, including frames whose corruption produces an empty detection list.

### Source GT versus evaluation GT

There are two representations of the same ground truth, not two independent annotations:

1. `labels_3d/<sequence>.json` is the source/runner representation. It is used to generate pseudo detections and only supplied to PedRefTrack in `gt_assisted` mode.
2. TrackEval consumes KITTI/JRDB text under `<gt_folder>/label_02/<sequence>.txt`, plus `evaluate_tracking.seqmap.<split>`. Generate it from the JSON with `tracker-eval-prepare-gt`.

The output of a tracker is not JSON ground truth; it is the same KITTI/JRDB text convention used by the evaluator. Preparing evaluation GT therefore converts the source JSON without changing the underlying annotations.

## Coordinate and export transforms

The important transforms are centralized and applied identically to detections and source GT:

| Stage | Operation | Implementation |
|---|---|---|
| local → global | `p_global = R(q) @ p_local + t` | `transform_box7_local_to_global` in `common/odometry_transform.py` |
| JRDB yaw compensation | `yaw_global = yaw_local - ego_yaw` | same function; this empirical sign is required by the existing JRDB pipeline |
| sequence transform | pose row selected by integer frame key | `load_odometry_csv`, `transform_frame_data_to_global`, `transform_sequence_to_global` |
| internal → TrackEval | `x=-cy`, `y=-cz+h/2`, `z=cx`, `w=w`, `h=h`, `d=l`, `yaw=(-rot_z) mod 2π` | `trackeval_xyzwhd_from_internal_center` in `export/jrdb_kitti_writer.py` |

The odometry CSV is treated as `T_world_sensor`. Global transformation happens in `runner/run_split.py` before a tracker step; `tracker-eval-prepare-gt --global-coords` applies the same transform when building global evaluation GT.

### Tracker output text

`write_sequence_kitti_txt` writes one object per line, sorted by `(frame, track_id)`:

```text
frame track_id class truncated occluded alpha x1 y1 x2 y2 x y z w h d yaw [score]
```

Columns 6–9 are dummy 2D boxes for 3D-only evaluation. Columns 10–16 are the TrackEval `xyzwhd` box after the mapping above. KITTI/JRDB output is mandatory because it is the input to every evaluation path. Duplicate IDs within a frame are rejected.

Tracker runs are saved as:

```text
<out_root>/<tracker_output_name>/<split_name>/
├── data/<sequence>.txt
├── frame_stats/<sequence>.csv
├── runtime_summary.csv
└── runtime_summary.json
```

Sequential mode records real per-frame step timings and is the mode to use for runtime measurements. Parallel mode runs sequences in a shared process pool for throughput; timing and per-frame profiling are intentionally disabled because concurrent workers make those measurements scientifically unreliable.

## Quick pipeline

Run one tracker sequentially (valid runtime statistics):

```bash
tracker-eval \
  --split_root /data/JRDB/test \
  --split_name test \
  --out_root /data/tracker_outputs \
  --trackers pedreftrack \
  --pedreftrack_modes no_gt \
  --global_coords \
  --odometry_root /data/JRDB/odometry
```

Run several trackers concurrently (throughput, not timing):

```bash
tracker-eval \
  --split_root /data/JRDB/test \
  --split_name test \
  --out_root /data/tracker_outputs \
  --trackers cbmot elptnet fastpoly gnnpmb simpletrack pedreftrack \
  --parallel --num_workers 12 \
  --global_coords --odometry_root /data/JRDB/odometry
```

Generate all clean/dropout/instability/combined pseudo detections:

```bash
tracker-eval-generate-pseudo \
  --split_root /data/JRDB/test \
  --split_name test \
  --odometry_root /data/JRDB/odometry \
  --spec tracker_eval/cli/pseudo_det_spec.yaml
```

Run trackers over the manifest conditions:

```bash
tracker-eval-pseudo \
  --manifest /data/JRDB/test/detections_3D_pseudo/manifest.json \
  --out_root /data/tracker_outputs \
  --trackers cbmot fastpoly pedreftrack \
  --pedreftrack_modes no_gt \
  --parallel --num_workers 12 \
  --global_coords
```

Pseudo runs use one canonical folder order: `<tracker>__global[_pedreftrack_mode]_<variant>`, for example `fastpoly__global_instability_L1` or `pedreftrack__global_no_gt_clean`. The protocol evaluator preserves these names in its result directory.

Prepare global evaluation GT and evaluate normal tracker outputs:

```bash
tracker-eval-prepare-gt \
  --labels-dir /data/JRDB/test/labels_3d \
  --gt-folder /data/eval_gt_global \
  --split test \
  --global-coords --odometry-root /data/JRDB/odometry

tracker-eval-evaluate \
  --gt-folder /data/eval_gt_global \
  --trackers-folder /data/tracker_outputs \
  --output-folder /data/evaluation_results \
  --split test
```

Evaluate all pseudo-detection conditions against that same GT:

```bash
tracker-eval-protocol pseudo \
  --trackers-dir /data/tracker_outputs \
  --gt-folder /data/eval_gt_global \
  --output-dir /data/pseudo_results \
  --split test --workers 12
```

Compute the tracker capability tables with one command:

```bash
tracker-eval-protocol capabilities \
  --trackers-base-dir /data/tracker_outputs \
  --gt-folder /data/eval_gt_global \
  --local-gt-folder /data/eval_gt_local \
  --detections-dir /data/JRDB/test/detections_3D \
  --output-dir /data/protocol_results \
  --trackers pedreftrack__global_gt_assisted,pedreftrack__global_no_gt,cbmot__global,fastpoly__global \
  --reference-tracker pedreftrack__global_gt_assisted \
  --split test --workers 8
```

This command builds/reuses common event caches, per-tracker capability/HOTA caches and final tables. The internal modules are `protocol/hota_cache.py`, `profiles.py` and `results.py`; users should normally call only `tracker-eval-protocol`.

The public command defaults to the corrected two-second RAL protocol:

- gap prediction is evaluated through `2.0` seconds;
- recovery bins are `1,4,7,10,13,16,19,22,25,28,31` frames;
- the remaining IoU, continuity, initialization, NN-bin, quantile and bootstrap defaults match the settings used for the corrected RAL tables.

`tables/initialization_latency_summary.csv` contains all initialization views in one run. Use `event_type == "trajectory_start"` for genuine trajectory initialization only, or `event_type == "all"` for trajectory initialization plus post-gap reacquisition when all evaluated trackers already lost the trajectory. The plotting notebook exposes this as `INITIALIZATION_EVENT_TYPE`; it does not require rerunning the protocol.

Changing a protocol-defining setting for an existing output directory is rejected to protect cached results. Use a new output directory, or pass `--force` when replacement of the old protocol cache is intentional.

Plot saved results by opening:

- `notebooks/plot_tracker_results.ipynb`
- `notebooks/plot_pseudo_detection_results.ipynb`
- `notebooks/ral_visualization_figures_jrdb.ipynb`

Set the path variables in the first configuration cell, then run all cells. A command-focused walkthrough is in [docs/PIPELINE.md](docs/PIPELINE.md).

## Repository map

```text
tracker_eval/                  runner, adapters, compact tracker runtimes
trackeval/                     corrected compact JRDB 3D evaluation runtime
tracker_eval/protocol/         reusable cache/profile/result stages
notebooks/                     result and manuscript plotting
third_party/                   TrackEval license and upstream notice
```

See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for upstream references. Project-authored code is MIT licensed; retained third-party files remain under their upstream terms.
