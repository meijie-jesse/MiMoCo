Place HDF5 datasets under `MIMOCO_DATA_DIR` (or the default repo `data/` subdirectory). Files must contain the keys expected by `utils.EpisodicDataset` (e.g. `/observations/qpos`, `/observations/qvel`, `/action`, `/observations/images/<camera>`). Camera keys must match `--camera_names` at training time.

