# AB3DMOT evaluation subset

This directory contains only the AB3DMOT files required by
`tracker_eval.trackers.ab3dmot_adapter`. It is based on the official
[xinshuoweng/AB3DMOT](https://github.com/xinshuoweng/AB3DMOT) implementation;
use the upstream repository for training, demos, visualization and the complete
project history.

The adapter adds this directory to `sys.path` automatically. No separate clone
or install is needed when this directory is present. Its runtime dependencies
are declared by the root `tracker-eval` package.

The repository-level `.gitignore` intentionally ignores
`local_trackers/ab3dmot/`, so the supplied local research copy is not staged by
Git accidentally.

The original AB3DMOT license is retained in `LICENSE`.
