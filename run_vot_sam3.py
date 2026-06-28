from __future__ import annotations

import argparse
import sys
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Optional

import numpy as np
import torch
from PIL import Image

from vot_mask_utils import (
    load_initial_masks,
    load_sequence_metadata,
    resolve_color_frames,
    write_object_results,
)

ROOT = Path(__file__).resolve().parent
SAM3_ROOT = ROOT / "sam3-main"
if str(SAM3_ROOT) not in sys.path:
    sys.path.insert(0, str(SAM3_ROOT))

from sam3.model_builder import build_tracker  # noqa: E402
from sam3.model.io_utils import load_resource_as_video_frames  # noqa: E402

DEFAULT_CHECKPOINT = Path(
    "/data/Disk_C/wanghe/vots2026_code/sam3/pre_model_pt/sam3.pt"
)
DEFAULT_DATASET_ROOT = Path("/data/Disk_C/wanghe/vots2025_votstval/sequences/")


def eprint(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


@dataclass
class Sam3SequenceSession:
    object_ids: List[int]
    image_shape: tuple[int, int]
    propagator: Optional[Iterator]
    done: bool = False


class Sam3VOTSequenceRunner:
    def __init__(
        self,
        checkpoint_path: Optional[Path],
        device: str = "cuda",
        compile_model: bool = False,
        offload_video_to_cpu: bool = False,
        offload_state_to_cpu: bool = False,
        debug_dir: Optional[Path] = None,
    ) -> None:
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for SAM3 tracker but is not available.")

        self.model = build_tracker(
            apply_temporal_disambiguation=True,
            with_backbone=True,
            compile_mode="default" if compile_model else None,
        )
        self.model = self.model.to(device=device).eval()
        if checkpoint_path:
            self._load_tracker_checkpoint(checkpoint_path)
        self.offload_video_to_cpu = offload_video_to_cpu
        self.offload_state_to_cpu = offload_state_to_cpu
        self.debug_dir = debug_dir
        if self.debug_dir is not None:
            self.debug_dir.mkdir(parents=True, exist_ok=True)

    def _load_tracker_checkpoint(self, checkpoint_path: Path) -> None:
        checkpoint = torch.load(str(checkpoint_path), map_location="cpu", weights_only=True)
        if isinstance(checkpoint, dict) and "model" in checkpoint:
            checkpoint = checkpoint["model"]
        if not isinstance(checkpoint, dict):
            raise RuntimeError(f"Unsupported checkpoint format: {checkpoint_path}")

        state_dict = self._extract_tracker_state_dict(checkpoint)
        missing_keys, unexpected_keys = self.model.load_state_dict(
            state_dict, strict=False
        )
        eprint(f"[checkpoint] loaded tracker weights from {checkpoint_path}")
        if unexpected_keys:
            eprint(f"[checkpoint] first unexpected keys: {unexpected_keys[:10]}")
        if missing_keys:
            eprint(f"[checkpoint] first missing keys: {missing_keys[:10]}")

    @staticmethod
    def _extract_tracker_state_dict(checkpoint: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        tracker_state: Dict[str, torch.Tensor] = {}

        if any(key.startswith("tracker.") for key in checkpoint):
            prefix = "tracker."
            tracker_state.update(
                {
                    key[len(prefix) :]: value
                    for key, value in checkpoint.items()
                    if key.startswith(prefix)
                }
            )

        if any(key.startswith("sam2_predictor.") for key in checkpoint):
            prefix = "sam2_predictor."
            tracker_state.update(
                {
                    key[len(prefix) :]: value
                    for key, value in checkpoint.items()
                    if key.startswith(prefix)
                }
            )

        if any(key.startswith("detector.backbone.") for key in checkpoint):
            prefix = "detector.backbone."
            tracker_state.update(
                {
                    f"backbone.{key[len(prefix):]}": value
                    for key, value in checkpoint.items()
                    if key.startswith(prefix)
                }
            )

        return tracker_state or checkpoint

    def run_sequence(
        self,
        sequence_dir: Path,
        initial_masks: Optional[Dict[int, np.ndarray]] = None,
        disable_progress: bool = False,
    ) -> Dict[int, List[np.ndarray]]:
        load_sequence_metadata(sequence_dir)
        frame_paths = resolve_color_frames(sequence_dir)
        if not frame_paths:
            raise RuntimeError(f"No frames found for sequence: {sequence_dir}")

        first_frame = Image.open(frame_paths[0]).convert("RGB")
        width, height = first_frame.size
        image_shape = (height, width)
        if initial_masks is None:
            initial_masks = load_initial_masks(sequence_dir, image_shape=image_shape)
        self._maybe_dump_debug_masks(
            sequence_dir=sequence_dir,
            frame_image=first_frame,
            masks=initial_masks,
            tag="decoded_init",
        )
        ordered_masks = OrderedDict(sorted(initial_masks.items()))
        object_ids = list(ordered_masks.keys())
        results = self._build_empty_results(
            object_ids=object_ids,
            num_frames=len(frame_paths),
            image_shape=image_shape,
        )

        session, frame0_predictions = self.start_sequence(
            sequence_dir=sequence_dir,
            initial_masks=initial_masks,
            disable_progress=disable_progress,
        )
        for object_id, mask in frame0_predictions.items():
            results[object_id][0] = mask
        self._maybe_dump_debug_predictions(
            sequence_dir=sequence_dir,
            frame_image=first_frame,
            predictions=results,
            frame_index=0,
            tag="pred_frame_00000001",
        )

        frame_index = 1
        while frame_index < len(frame_paths):
            frame_predictions = self.next_frame(session, sequence_dir.name)
            for object_id, mask in frame_predictions.items():
                results[object_id][frame_index] = mask
            frame_image = Image.open(frame_paths[frame_index]).convert("RGB")
            self._maybe_dump_debug_predictions(
                sequence_dir=sequence_dir,
                frame_image=frame_image,
                predictions=results,
                frame_index=frame_index,
                tag=f"pred_frame_{frame_index + 1:08d}",
            )
            frame_index += 1

        return results

    def start_sequence(
        self,
        sequence_dir: Path,
        initial_masks: Dict[int, np.ndarray],
        disable_progress: bool = False,
    ) -> tuple[Sam3SequenceSession, Dict[int, np.ndarray]]:
        frame_paths = resolve_color_frames(sequence_dir)
        if not frame_paths:
            raise RuntimeError(f"No frames found for sequence: {sequence_dir}")
        eprint(
            f"[diag] start_sequence sequence_dir={sequence_dir} frame_dir={frame_paths[0].parent} num_frames={len(frame_paths)}"
        )

        ordered_masks = OrderedDict(sorted(initial_masks.items()))
        object_ids = list(ordered_masks.keys())
        first_mask = next(iter(ordered_masks.values()))
        image_shape = first_mask.shape
        frame0_predictions = self._empty_frame_predictions(object_ids, image_shape)

        try:
            images, video_height, video_width = self._load_frames_for_vot(frame_paths)
            inference_state = self.model.init_state(
                video_height=video_height,
                video_width=video_width,
                num_frames=len(frame_paths),
                offload_video_to_cpu=self.offload_video_to_cpu,
                offload_state_to_cpu=self.offload_state_to_cpu,
            )
            inference_state["images"] = images
        except Exception as exc:
            eprint(f"[warn] init_state failed for {sequence_dir.name}: {exc}")
            return Sam3SequenceSession(object_ids, image_shape, None, done=True), frame0_predictions

        for object_id in object_ids:
            try:
                frame_index, returned_ids, _, video_res_masks = self.model.add_new_mask(
                    inference_state=inference_state,
                    frame_idx=0,
                    obj_id=object_id,
                    mask=torch.from_numpy(ordered_masks[object_id].astype(np.uint8)),
                    add_mask_to_memory=False,
                )
                if frame_index == 0:
                    self._store_frame_predictions_dict(
                        frame0_predictions, returned_ids, video_res_masks
                    )
            except Exception as exc:
                eprint(
                    f"[warn] add_new_mask failed for {sequence_dir.name} object {object_id}: {exc}"
                )

        try:
            propagator = iter(
                self.model.propagate_in_video(
                    inference_state=inference_state,
                    start_frame_idx=0,
                    max_frame_num_to_track=None,
                    reverse=False,
                    tqdm_disable=disable_progress,
                    propagate_preflight=True,
                )
            )
        except Exception as exc:
            eprint(f"[warn] propagate_in_video setup failed for {sequence_dir.name}: {exc}")
            propagator = None

        return Sam3SequenceSession(object_ids, image_shape, propagator), frame0_predictions

    def _load_frames_for_vot(self, frame_paths: List[Path]):
        pil_images = []
        try:
            for frame_path in frame_paths:
                pil_images.append(Image.open(frame_path).convert("RGB"))
            return load_resource_as_video_frames(
                resource_path=pil_images,
                image_size=self.model.image_size,
                offload_video_to_cpu=self.offload_video_to_cpu,
            )
        except Exception as exc:
            raise RuntimeError(
                f"failed to preload VOT frames from {frame_paths[0].parent}: {exc}"
            ) from exc

    def next_frame(
        self,
        session: Sam3SequenceSession,
        sequence_name: str,
        return_scores: bool = False,
    ) -> Dict[int, np.ndarray]:
        frame_predictions = self._empty_frame_predictions(
            session.object_ids,
            session.image_shape,
            dtype=np.float32 if return_scores else np.uint8,
        )
        if session.done or session.propagator is None:
            session.done = True
            return frame_predictions

        while True:
            try:
                (
                    frame_index,
                    returned_ids,
                    _low_res_masks,
                    video_res_masks,
                    _scores,
                ) = next(session.propagator)
            except StopIteration:
                session.done = True
                return frame_predictions
            except Exception as exc:
                eprint(f"[warn] propagate_in_video failed for {sequence_name}: {exc}")
                session.done = True
                return frame_predictions

            if frame_index <= 0:
                continue

            self._store_frame_predictions_dict(
                frame_predictions,
                returned_ids,
                video_res_masks,
                binarize=not return_scores,
            )
            return frame_predictions

    @staticmethod
    def _build_empty_results(
        object_ids: List[int],
        num_frames: int,
        image_shape: tuple[int, int],
    ) -> Dict[int, List[np.ndarray]]:
        return {
            object_id: [np.zeros(image_shape, dtype=np.uint8) for _ in range(num_frames)]
            for object_id in object_ids
        }

    @staticmethod
    def _store_frame_predictions(
        results: Dict[int, List[np.ndarray]],
        frame_index: int,
        object_ids: List[int],
        video_res_masks: torch.Tensor,
    ) -> None:
        masks_np = video_res_masks.detach().cpu().numpy()
        for idx, object_id in enumerate(object_ids):
            mask = np.asarray(masks_np[idx])
            while mask.ndim > 2:
                mask = np.squeeze(mask, axis=0)
            results[int(object_id)][frame_index] = (mask > 0).astype(np.uint8)

    @staticmethod
    def _empty_frame_predictions(
        object_ids: List[int], image_shape: tuple[int, int], dtype=np.uint8
    ) -> Dict[int, np.ndarray]:
        return {
            object_id: np.zeros(image_shape, dtype=dtype) for object_id in object_ids
        }

    @staticmethod
    def _store_frame_predictions_dict(
        frame_predictions: Dict[int, np.ndarray],
        object_ids: List[int],
        video_res_masks: torch.Tensor,
        binarize: bool = True,
    ) -> None:
        masks_np = video_res_masks.detach().cpu().numpy()
        for idx, object_id in enumerate(object_ids):
            if int(object_id) not in frame_predictions:
                continue
            mask = np.asarray(masks_np[idx], dtype=np.float32)
            while mask.ndim > 2:
                mask = np.squeeze(mask, axis=0)
            if binarize:
                frame_predictions[int(object_id)] = (mask > 0).astype(np.uint8)
            else:
                frame_predictions[int(object_id)] = mask


    def _maybe_dump_debug_masks(
        self,
        sequence_dir: Path,
        frame_image: Image.Image,
        masks: Dict[int, np.ndarray],
        tag: str,
    ) -> None:
        if self.debug_dir is None:
            return
        image_np = np.array(frame_image, dtype=np.uint8)
        sequence_debug_dir = self.debug_dir / sequence_dir.name
        sequence_debug_dir.mkdir(parents=True, exist_ok=True)
        self._save_overlay(
            image_np=image_np,
            masks=masks,
            output_path=sequence_debug_dir / f"{tag}.png",
        )

    def _maybe_dump_debug_predictions(
        self,
        sequence_dir: Path,
        frame_image: Image.Image,
        predictions: Dict[int, List[np.ndarray]],
        frame_index: int,
        tag: str,
    ) -> None:
        if self.debug_dir is None:
            return
        image_np = np.array(frame_image, dtype=np.uint8)
        sequence_debug_dir = self.debug_dir / sequence_dir.name
        sequence_debug_dir.mkdir(parents=True, exist_ok=True)
        masks = {
            object_id: object_predictions[frame_index]
            for object_id, object_predictions in predictions.items()
        }
        self._save_overlay(
            image_np=image_np,
            masks=masks,
            output_path=sequence_debug_dir / f"{tag}.png",
        )

    @staticmethod
    def _save_overlay(
        image_np: np.ndarray,
        masks: Dict[int, np.ndarray],
        output_path: Path,
    ) -> None:
        colors = [
            np.array([255, 0, 0], dtype=np.uint8),
            np.array([0, 255, 255], dtype=np.uint8),
            np.array([0, 255, 0], dtype=np.uint8),
            np.array([255, 255, 0], dtype=np.uint8),
        ]
        overlay = image_np.copy()
        for index, object_id in enumerate(sorted(masks)):
            mask = np.asarray(masks[object_id], dtype=bool)
            color = colors[index % len(colors)]
            overlay[mask] = (0.55 * overlay[mask] + 0.45 * color).astype(np.uint8)
            border = mask & (
                ~(
                    np.roll(mask, 1, 0)
                    & np.roll(mask, -1, 0)
                    & np.roll(mask, 1, 1)
                    & np.roll(mask, -1, 1)
                )
            )
            overlay[border] = color
        Image.fromarray(overlay).save(output_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run SAM3 on VOT/VOTSt sequences and export VOT mask files."
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=DEFAULT_DATASET_ROOT,
        help="Root directory that contains list.txt and sequence subdirectories.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default="./output_VOTST",
        help="Output directory. Results are written to baseline/<sequence>/ by default.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT,
        help="Local SAM3 tracker checkpoint.",
    )
    parser.add_argument(
        "--sequence",
        nargs="*",
        default=None,
        help="Optional subset of sequence names. Default: use list.txt or all subdirs.",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        choices=["cuda"],
        help="Inference device. SAM3 tracker currently expects CUDA.",
    )
    parser.add_argument(
        "--compile",
        action="store_true",
        help="Enable tracker compilation where supported.",
    )
    parser.add_argument(
        "--filename-template",
        default="{sequence}_object{object_id}.txt",
        help="Filename template for each object's exported result file.",
    )
    parser.add_argument(
        "--no-tqdm",
        action="store_true",
        help="Disable propagation progress output.",
    )
    parser.add_argument(
        "--debug-dir",default="./debug_vis",
        type=Path,
        help="Optional directory to dump init-mask and first-frame prediction overlays.",
    )
    return parser.parse_args()


def discover_sequences(dataset_root: Path, requested: Optional[List[str]]) -> List[Path]:
    if requested:
        return [dataset_root / name for name in requested]

    list_file = dataset_root / "list.txt"
    if list_file.is_file():
        names = [
            line.strip()
            for line in list_file.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        return [dataset_root / name for name in names]

    return sorted(p for p in dataset_root.iterdir() if p.is_dir())


def main() -> None:
    args = parse_args()
    runner = Sam3VOTSequenceRunner(
        checkpoint_path=args.checkpoint,
        device=args.device,
        compile_model=args.compile,
        debug_dir=args.debug_dir,
    )

    sequence_dirs = discover_sequences(args.dataset_root, args.sequence)
    if not sequence_dirs:
        raise RuntimeError(f"No sequences found under {args.dataset_root}")

    for sequence_dir in sequence_dirs:
        sequence_name = sequence_dir.name
        predictions = runner.run_sequence(
            sequence_dir, disable_progress=args.no_tqdm
        )
        out_dir = args.output_root / "baseline" / sequence_name
        written = write_object_results(
            output_dir=out_dir,
            sequence_name=sequence_name,
            object_predictions=predictions,
            filename_template=args.filename_template,
        )
        eprint(f"[OK] {sequence_name}: wrote {len(written)} files to {out_dir}")


if __name__ == "__main__":
    main()
