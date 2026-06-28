from __future__ import annotations
import argparse
import sys
import traceback
from pathlib import Path
from typing import Dict, List
from urllib.parse import unquote, urlparse
import numpy as np
import torch
from PIL import Image
from run_vot_sam3 import DEFAULT_CHECKPOINT, Sam3VOTSequenceRunner
# from vot_mask_utils import decode_mask, load_sequence_metadata
from vot_mask_utils import decode_mask, load_initial_masks, load_sequence_metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="VOT toolkit wrapper for SAM3 mask-initialized video tracking."
    )
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--device", default="cuda", choices=["cuda"])
    parser.add_argument(
        "--offload-video-to-cpu",
        action="store_true",
        default=True,
        help="Keep decoded video frames on CPU to reduce GPU memory pressure.",
    )
    parser.add_argument(
        "--offload-state-to-cpu",
        action="store_true",
        help="Offload inference state to CPU to reduce GPU memory pressure.",
    )
    return parser.parse_args()


def log_cuda_diagnostics() -> None:
    if not torch.cuda.is_available():
        print("[diag] CUDA is not available", file=sys.stderr, flush=True)
        return

    try:
        device_index = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(device_index)
        allocated = torch.cuda.memory_allocated(device_index) / (1024 ** 3)
        reserved = torch.cuda.memory_reserved(device_index) / (1024 ** 3)
        total = props.total_memory / (1024 ** 3)
        print(
            (
                f"[diag] cuda_device={device_index} "
                f"name={props.name} total_gb={total:.2f} "
                f"allocated_gb={allocated:.2f} reserved_gb={reserved:.2f}"
            ),
            file=sys.stderr,
            flush=True,
        )
    except Exception as exc:
        print(f"[diag] failed to query CUDA diagnostics: {exc}", file=sys.stderr, flush=True)


def make_full_size(mask: np.ndarray, image_shape: tuple[int, int]) -> np.ndarray:
    height, width = image_shape
    if mask.shape == image_shape:
        return (mask > 0).astype(np.uint8)

    full = np.zeros(image_shape, dtype=np.uint8)
    copy_h = min(height, mask.shape[0])
    copy_w = min(width, mask.shape[1])
    full[:copy_h, :copy_w] = mask[:copy_h, :copy_w] > 0
    return full


def region_to_mask(region: object, image_shape: tuple[int, int]) -> np.ndarray:
    if isinstance(region, np.ndarray):
        return make_full_size(np.asarray(region, dtype=np.uint8), image_shape)
    if hasattr(region, "mask") and hasattr(region, "offset"):
        cropped = np.asarray(region.mask, dtype=np.uint8)
        x0, y0 = region.offset
        full = np.zeros(image_shape, dtype=np.uint8)
        y0 = max(int(y0), 0)
        x0 = max(int(x0), 0)
        y1 = min(y0 + cropped.shape[0], image_shape[0])
        x1 = min(x0 + cropped.shape[1], image_shape[1])
        if y1 > y0 and x1 > x0:
            full[y0:y1, x0:x1] = cropped[: y1 - y0, : x1 - x0] > 0
        return full
    return decode_mask(str(region), image_shape=image_shape)


def normalize_frame_path(frame: str) -> Path:
    parsed = urlparse(frame)
    if parsed.scheme == "file":
        return Path(unquote(parsed.path)).resolve()
    return Path(frame).resolve()


def infer_sequence_dir(first_frame_path: Path) -> Path:
    candidates = []
    if first_frame_path.parent.name.lower() == "color":
        candidates.append(first_frame_path.parent.parent)
    candidates.extend([first_frame_path.parent, first_frame_path.parent.parent])

    for candidate in candidates:
        if (candidate / "sequence").is_file():
            return candidate
    for candidate in candidates:
        color_dir = candidate / "color"
        if color_dir.is_dir():
            return candidate
    raise RuntimeError(
        f"Could not infer VOT sequence directory from frame path: {first_frame_path}"
    )


def scores_to_regions(
    frame_predictions: Dict[int, np.ndarray], object_ids: List[int]
) -> List[object]:
    if not object_ids:
        return []

    image_shape = np.asarray(frame_predictions[object_ids[0]]).shape
    score_stack = np.full((len(object_ids), *image_shape), -1e10, dtype=np.float32)
    for idx, object_id in enumerate(object_ids):
        score = np.asarray(frame_predictions.get(object_id), dtype=np.float32)
        while score.ndim > 2:
            score = np.squeeze(score, axis=0)
        score_stack[idx] = score

    best_indices = np.argmax(score_stack, axis=0)
    best_scores = np.take_along_axis(score_stack, best_indices[None, ...], axis=0)[0]
    merged_mask = np.zeros(image_shape, dtype=np.uint8)
    foreground = best_scores > 0
    merged_mask[foreground] = best_indices[foreground].astype(np.uint8) + 1

    import vot

    regions: List[object] = []
    for idx, object_id in enumerate(object_ids, start=1):
        mask = (merged_mask == idx).astype(np.uint8)
        if np.sum(mask) == 0:
            regions.append(None)
        else:
            regions.append(mask)
    return regions


def mask_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    mask_a = np.asarray(mask_a, dtype=bool)
    mask_b = np.asarray(mask_b, dtype=bool)
    union = np.logical_or(mask_a, mask_b).sum()
    if union == 0:
        return 1.0
    intersection = np.logical_and(mask_a, mask_b).sum()
    return float(intersection) / float(union)


def align_masks_to_trax_order(
    trax_initial_masks: Dict[int, np.ndarray],
    gt_initial_masks: Dict[int, np.ndarray],
) -> Dict[int, np.ndarray]:
    if not trax_initial_masks:
        return dict(gt_initial_masks)
    if len(trax_initial_masks) != len(gt_initial_masks):
        return dict(gt_initial_masks)

    remaining_gt_ids = set(gt_initial_masks)
    aligned_masks: Dict[int, np.ndarray] = {}

    for trax_id in sorted(trax_initial_masks):
        best_gt_id = None
        best_iou = -1.0
        for gt_id in sorted(remaining_gt_ids):
            score = mask_iou(trax_initial_masks[trax_id], gt_initial_masks[gt_id])
            if score > best_iou:
                best_iou = score
                best_gt_id = gt_id
        if best_gt_id is None:
            continue
        aligned_masks[trax_id] = gt_initial_masks[best_gt_id]
        remaining_gt_ids.remove(best_gt_id)

    if len(aligned_masks) != len(trax_initial_masks):
        return dict(gt_initial_masks)
    return aligned_masks


def main() -> None:
    args = parse_args()

    import vot

    handle = vot.VOT("mask", multiobject=True)
    objects = handle.objects()
    first_frame = handle.frame()
    if not first_frame:
        raise RuntimeError("VOT did not provide the initialization frame.")

    first_frame_path = normalize_frame_path(first_frame)
    sequence_dir = infer_sequence_dir(first_frame_path)
    print(
        f"[diag] first_frame={first_frame} resolved={first_frame_path} sequence_dir={sequence_dir}",
        file=sys.stderr,
        flush=True,
    )
    image = Image.open(first_frame_path).convert("RGB")
    width, height = image.size
    image_shape = (height, width)

    # initial_masks: Dict[int, np.ndarray] = {
    #     idx: region_to_mask(region, image_shape)
    #     for idx, region in enumerate(objects, start=1)
    # }
    # object_ids = sorted(initial_masks)
    trax_initial_masks: Dict[int, np.ndarray] = {
        idx: region_to_mask(region, image_shape)
        for idx, region in enumerate(objects, start=1)
    }
    gt_initial_masks = load_initial_masks(sequence_dir, image_shape)
    initial_masks = align_masks_to_trax_order(trax_initial_masks, gt_initial_masks)
    object_ids = sorted(initial_masks)

    if len(trax_initial_masks) != len(gt_initial_masks):
        print(
            (
                f"[diag] object_count_mismatch trax={len(trax_initial_masks)} "
                f"groundtruth={len(gt_initial_masks)}"
            ),
            file=sys.stderr,
            flush=True,
        )

    overlap_ids = sorted(set(trax_initial_masks).intersection(initial_masks))
    if overlap_ids:
        diffs = []
        for object_id in overlap_ids:
            trax_area = int(np.sum(trax_initial_masks[object_id] > 0))
            gt_area = int(np.sum(initial_masks[object_id] > 0))
            xor_area = int(
                np.sum(
                    np.logical_xor(
                        trax_initial_masks[object_id] > 0,
                        initial_masks[object_id] > 0,
                    )
                )
            )
            diffs.append(
                f"obj{object_id}:trax={trax_area},gt={gt_area},xor={xor_area}"
            )
        print(
            "[diag] init_mask_compare " + " | ".join(diffs),
            file=sys.stderr,
            flush=True,
        )

    if len(trax_initial_masks) == len(gt_initial_masks):
        mapping_parts = []
        used_gt_ids = set()
        for trax_id in sorted(trax_initial_masks):
            matched_gt_id = None
            for gt_id in sorted(gt_initial_masks):
                if gt_id in used_gt_ids:
                    continue
                if initial_masks.get(trax_id) is gt_initial_masks[gt_id]:
                    matched_gt_id = gt_id
                    used_gt_ids.add(gt_id)
                    break
            if matched_gt_id is not None:
                mapping_parts.append(f"trax{trax_id}->gt{matched_gt_id}")
        if mapping_parts:
            print(
                "[diag] init_mask_order " + " | ".join(mapping_parts),
                file=sys.stderr,
                flush=True,
            )










    runner = Sam3VOTSequenceRunner(
        checkpoint_path=args.checkpoint,
        device=args.device,
        compile_model=args.compile,
        offload_video_to_cpu=args.offload_video_to_cpu,
        offload_state_to_cpu=args.offload_state_to_cpu,
    )
    session, _frame0_predictions = runner.start_sequence(
        sequence_dir, initial_masks=initial_masks, disable_progress=True
    )

    while True:
        imagefile = handle.frame()
        if not imagefile:
            break

        frame_predictions = runner.next_frame(
            session, sequence_dir.name, return_scores=True
        )
        regions = scores_to_regions(frame_predictions, object_ids)
        handle.report(regions)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[fatal] tracker crashed: {exc}", file=sys.stderr, flush=True)
        traceback.print_exc(file=sys.stderr)
        log_cuda_diagnostics()
        raise
