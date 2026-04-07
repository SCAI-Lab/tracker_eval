# tracker_eval

Tracker-only inference, export, and runtime-profiling pipeline for **CROWDTRACKBENCH**, a benchmark for **3D LiDAR pedestrian multi-object tracking**.

This repository was used for the experiments in our IROS 2026 benchmark paper. Its role is intentionally focused: it runs trackers on a shared set of per-frame 3D detections, measures tracker-side computational performance, and exports predictions in a format that is directly consumable by the official JRDB evaluation toolkit.

## What this repository is for

`tracker_eval` is designed to evaluate the **tracking stage only** under a shared-detection protocol. In the benchmark setup, all trackers consume the same per-frame 3D pedestrian detections so that differences in results are driven by data association, motion handling, and track lifecycle logic rather than by detector changes.

Concretely, this repository is used to:

- run multiple 3D pedestrian trackers on JRDB detections,
- profile tracker-step runtime (for example FPS and per-frame latency statistics),
- export tracker outputs to **JRDB3DBox-compatible KITTI-tracking `.txt` files**, and
- support controlled stress tests via GT-derived pseudo detections.

This repository is **not** the official accuracy evaluator. For final tracking metrics, we use the official JRDB toolkit:

- `jrdb_toolkit/tracking_eval`: <https://github.com/JRDB-dataset/jrdb_toolkit/tree/main/tracking_eval>

## Benchmark context

The accompanying benchmark paper introduces **CROWDTRACKBENCH** as a reproducible tracker-only benchmark for **3D pedestrian MOT on JRDB** with shared detections, scenario-based analysis, controlled pseudo-detection stress tests, and embedded tracker-step profiling. The primary benchmark metric used in the paper is **HOTA**, while identity stability is analyzed through ID switches and related statistics.

In the paper setup, the benchmark is run on **JRDB**, using shared 3D pedestrian detections and a common export/evaluation convention. This repository implements the tracking, export, and runtime-measurement side of that pipeline.

## Scope of this codebase

At a high level, the pipeline is:

1. Load per-sequence JRDB detections from JSON.
2. Run one tracker frame by frame.
3. Enforce evaluation-friendly output conventions such as unique track IDs per frame.
4. Save predicted trajectories in JRDB3DBox-compatible KITTI-tracking text format.
5. Save runtime summaries and per-frame timing statistics.

Optional utilities additionally:

- convert JRDB ground truth labels to the same KITTI-style convention,
- generate GT-derived pseudo detections for robustness studies,
- build TP/FP score distributions from detections and GT, and
- visualize predicted tracks against GT as videos.

## Supported trackers

The repository currently supports the following trackers:

- **Headroom**: an in-repo GT-assisted diagnostic reference tracker used to estimate remaining headroom under fixed detections.
- **AB3DMOT**
- **FastPoly**
- **GNN-PMB Tracker**
- **SimpleTrack**
- **CBMOT**
- **ELPTNet** (box-only variant used in the benchmark)

`Headroom` is the only tracker implemented directly in this repository. The other methods are integrated through lightweight adapters that wrap their original open-source implementations into a common tracker interface.

Upstream tracker repositories used in this benchmark:

- AB3DMOT: <https://github.com/xinshuoweng/AB3DMOT>
- FastPoly: <https://github.com/lixiaoyu2000/FastPoly>
- GNN-PMB Tracker: <https://github.com/chisyliu/GnnPmbTracker>
- SimpleTrack: <https://github.com/tusen-ai/SimpleTrack>
- CBMOT: <https://github.com/cogsys-tuebingen/CBMOT>
- ELPTNet: <https://github.com/jinzhengguang/ELPTNet>

## Repository structure

```text
tracker_eval/
├── cli/
│   ├── run_tracker.py
│   ├── run_all_trackers.py
│   ├── convert_gt_to_kitti_3d.py
│   ├── generate_pseudo_detections_from_gt.py
│   ├── build_score_distributions_from_gt_det.py
│   └── viz_tracks.py
├── runner/
│   ├── run_sequence.py
│   └── run_split.py
├── data/
│   └── jrdb_io.py
├── common/
│   ├── types.py
│   └── odometry_transform.py
├── export/
│   └── jrdb_kitti_writer.py
├── trackers/
│   ├── base.py
│   ├── headroom_adapter.py
│   ├── ab3dmot_adapter.py
│   ├── fastpoly_adapter.py
│   ├── gnnpmbtracker_adapter.py
│   ├── simpletrack_adapter.py
│   ├── cbmot_adapter.py
│   ├── elptnet_adapter.py
│   └── headroom_kf_adapter.py
└── utils.py
```

### Main modules

- **`trackers/base.py`**  
  Defines the common tracker interface used throughout the benchmark. Each tracker is reset per sequence and stepped frame by frame, while timing is recorded in a consistent way.

- **`runner/run_sequence.py`**  
  Core per-sequence execution logic. Runs one tracker on one sequence, computes runtime statistics, and converts outputs into exportable track rows.

- **`runner/run_split.py`**  
  Runs a tracker over a full split, writes KITTI-style outputs, saves per-sequence and aggregate runtime summaries, and optionally supports parallel execution across sequences.

- **`export/jrdb_kitti_writer.py`**  
  Converts the repository’s internal box convention into the JRDB3DBox / KITTI-style tracking format expected by the official toolkit.

- **`cli/run_tracker.py`**  
  Main entry point for running a single tracker over one or more JRDB splits.

- **`cli/run_all_trackers.py`**  
  Convenience wrapper for benchmarking several trackers in one pass.

- **`cli/convert_gt_to_kitti_3d.py`**  
  Converts JRDB `labels_3d` JSON files into evaluation-ready KITTI-tracking text files.

- **`cli/generate_pseudo_detections_from_gt.py`**  
  Generates GT-derived pseudo detections for controlled stress tests such as dropout, instability, and confuser cases.

- **`cli/build_score_distributions_from_gt_det.py`**  
  Builds TP/FP score distributions by matching detections to GT; these can be reused when sampling realistic pseudo-detection scores.

- **`cli/viz_tracks.py`**  
  Visualizes exported predictions and GT in XY/XZ/YZ views and renders MP4 videos.

## Expected data layout

The code assumes a JRDB-style split structure such as:

```text
<split_root>/
├── detections_3D/
│   ├── <sequence>.json
│   └── ...
└── labels_3d/
    ├── <sequence>.json
    └── ...
```

If global-coordinate evaluation is used, odometry is expected under:

```text
<odometry_root>/<split_name>/odometry/<sequence>.csv
```

## Output layout

Typical outputs are written under:

```text
<out_root>/
└── <tracker_name>/
    └── <split_name>/
        ├── data/
        │   ├── <sequence>.txt
        │   └── ...
        ├── frame_stats/
        │   ├── <sequence>.csv
        │   └── ...
        ├── runtime_summary.json
        └── runtime_summary.csv
```

Where:

- `data/*.txt` are the JRDB3DBox-compatible tracking results,
- `frame_stats/*.csv` store per-frame runtime and load information, and
- `runtime_summary.*` store aggregate sequence and split-level runtime statistics.

## Typical workflows

### 1. Run one tracker

```bash
python -m tracker_eval.cli.run_tracker \
  --split_root /path/to/JRDB/test \
  --split_name test \
  --out_root /path/to/outputs \
  --tracker ab3dmot
```

For wrapped trackers, additional tracker-specific configuration files may be required, for example:

- `--simpletrack_config`
- `--fastpoly_config`
- `--gnnpmb_parameters_path`

Run `--help` for the full list of tracker-specific arguments.

### 2. Run all trackers

```bash
python -m tracker_eval.cli.run_all_trackers \
  --split_root /path/to/JRDB/test \
  --split_name test \
  --out_root /path/to/outputs
```

### 3. Export ground truth in evaluation format

```bash
python -m tracker_eval.cli.convert_gt_to_kitti_3d \
  --split_root /path/to/JRDB/test \
  --split_name test \
  --out_root /path/to/outputs
```

### 4. Generate GT-derived pseudo detections

```bash
python -m tracker_eval.cli.generate_pseudo_detections_from_gt \
  --split_root /path/to/JRDB/test \
  --spec /path/to/pseudo_detection_spec.yaml
```

### 5. Build score distributions for pseudo detections

```bash
python -m tracker_eval.cli.build_score_distributions_from_gt_det \
  --dataset_root /path/to/JRDB \
  --out_dir /path/to/score_distributions
```

### 6. Visualize predictions vs. GT

```bash
python -m tracker_eval.cli.viz_tracks \
  --out_root /path/to/outputs \
  --tracker ab3dmot \
  --split_name test \
  --sequence bytes-cafe-2019-02-07_0 \
  --out_dir /path/to/videos
```

## Notes on evaluation

This repository exports predictions in the convention expected by the official JRDB tracking evaluator, but it does **not** replace the evaluator itself. The intended workflow is:

1. run tracker inference here,
2. export predictions to KITTI-style JRDB3DBox files,
3. run the official JRDB toolkit for final accuracy metrics.

This separation keeps the repository focused on:

- fair tracker-side comparison under shared detections,
- reproducible runtime profiling, and
- clean handoff to the official evaluation pipeline.

## Notes on implementation

- The repository uses a **common tracker interface** so different trackers can be benchmarked through the same runner.
- Outputs are validated to satisfy **unique track IDs per frame**, which is required by TrackEval / JRDB evaluation.
- `Headroom` supports GT-assisted tracking logic for diagnostic analysis, while the other trackers are primarily wrapped through adapter classes.
- Parallel execution is supported for throughput, but detailed timing and per-frame profiling are intentionally disabled in parallel mode.

## Summary

In short, `tracker_eval` is the repository that powers the **tracker inference and export side of CROWDTRACKBENCH**. It standardizes how multiple open-source 3D pedestrian trackers are run on JRDB detections, how their runtime is measured, and how their outputs are exported for official evaluation.