# SAM3 VOT/VOTSt runner

This workspace now contains two entry points:

- `run_vot_sam3.py`: offline runner for a dataset root with `list.txt` and per-sequence folders.
- `vot_tracker_sam3.py`: VOT toolkit wrapper for `traxpython`.

## Expected sequence format

Each sequence directory should look like:

```text
<sequence_name>/
  color/
    00000001.jpg
    00000002.jpg
    ...
  groundtruth_object1.txt
  groundtruth_object2.txt
  groundtruth__ignore.txt
  sequence
```

`groundtruth_object*.txt` is expected to use the VOT mask text format.
Only the first line is used as initialization. The exported prediction files also
use the same VOT mask format, one line per frame.

## Offline export

```bash
python run_vot_sam3.py \
  --dataset-root /data/Disk_C/wanghe/votst2026/sequences \
  --output-root /data/Disk_C/wanghe/vot_workspace/results/SAM3VOTST2026 \
  --checkpoint /data/Disk_C/wanghe/vots2026_code/sam3/pre_model_pt/sam3.pt
```

This writes:

```text
<output-root>/
  baseline/
    <sequence_name>/
      <sequence_name>_object1.txt
      <sequence_name>_object2.txt
      ...
```

## VOT toolkit integration

1. Put the official VOT python integration (`vot.py`) on the Python path used by the tracker.
2. Add an entry like `trackers.example.ini` into your workspace `trackers.ini`.
3. Run the official workspace flow, for example:


```bash
vot initialize vots2025/votstval --workspace /data/Disk_C/wanghe/vot_workspace
vot evaluate SAM3VOTST2026 --workspace /data/Disk_C/wanghe/vot_workspace
vot analysis SAM3VOTST2026 --workspace /data/Disk_C/wanghe/vot_workspace
vot pack SAM3VOTST2026 --workspace /data/Disk_C/wanghe/vot_workspace
```

## Notes

- The tracker currently precomputes sequence predictions before sending the
  first TraX response, so `trackers.ini` should use a generous timeout such as
  `timeout = 600000` rather than 10 seconds.
- The code now uses the SAM3 tracker path and initializes each object from the
  first-frame mask before running temporal propagation.
- `vot_tracker_sam3.py` auto-detects the sequence directory from the frame path
  provided by the toolkit.
- If you run the transformative single-object challenge, the same code works with
  only one `groundtruth_object*.txt` file.
- `run_vot_sam3.py` now defaults to your dataset root and checkpoint path, so you
  only need to supply `--output-root` during offline export.
