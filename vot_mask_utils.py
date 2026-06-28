from __future__ import annotations

import configparser
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np


def load_sequence_metadata(sequence_dir: Path) -> Dict[str, str]:
    sequence_file = sequence_dir / "sequence"
    if not sequence_file.is_file():
        raise FileNotFoundError(f"Missing sequence metadata file: {sequence_file}")

    parser = configparser.ConfigParser(interpolation=None)
    parser.read_string("[sequence]\n" + sequence_file.read_text(encoding="utf-8"))
    return dict(parser["sequence"])


def resolve_color_frames(sequence_dir: Path) -> List[Path]:
    color_dir: Optional[Path] = None
    sequence_file = sequence_dir / "sequence"
    if sequence_file.is_file():
        metadata = load_sequence_metadata(sequence_dir)
        color_pattern = metadata.get("channels.color", "color/%08d.jpg")
        color_dir = sequence_dir / Path(color_pattern).parent
    elif (sequence_dir / "color").is_dir():
        color_dir = sequence_dir / "color"

    if color_dir.is_dir():
        frames = sorted(
            p
            for p in color_dir.iterdir()
            if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
        )
        if frames:
            return frames

    # Some local debug/test sequences keep frames directly in the sequence root.
    frames = sorted(
        p for p in sequence_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    if frames:
        return frames

    raise FileNotFoundError(
        f"Missing frames under both {color_dir} and {sequence_dir}"
    )


def mask_to_rle(mask: np.ndarray, max_stride: int = 1 << 30) -> List[int]:
    flat = np.asarray(mask, dtype=np.uint8).reshape(-1)
    if flat.size == 0:
        return []

    rle: List[int] = []
    current = 0
    run = 0
    for value in flat:
        value = 1 if value else 0
        if value == current:
            run += 1
            if run == max_stride:
                rle.append(run)
                run = 0
        else:
            rle.append(run)
            current = value
            run = 1
    rle.append(run)
    return rle


def rle_to_mask(rle: Iterable[int], width: int, height: int) -> np.ndarray:
    total = width * height
    flat = np.zeros(total, dtype=np.uint8)
    index = 0
    value = 0
    for run in rle:
        run = int(run)
        if run < 0:
            raise ValueError(f"Invalid negative RLE run: {run}")
        if value == 1 and run > 0:
            flat[index : index + run] = 1
        index += run
        value = 1 - value
    if index != total:
        raise ValueError(
            f"RLE length mismatch: decoded {index} pixels for a {width}x{height} mask"
        )
    return flat.reshape(height, width)


def encode_mask(mask: np.ndarray) -> str:
    mask = np.asarray(mask, dtype=np.uint8)
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return "0"

    x0 = int(xs.min())
    y0 = int(ys.min())
    x1 = int(xs.max()) + 1
    y1 = int(ys.max()) + 1
    cropped = mask[y0:y1, x0:x1]
    runs = ",".join(str(v) for v in mask_to_rle(cropped))
    return f"m{x0},{y0},{x1 - x0},{y1 - y0},{runs}"


def decode_mask(
    encoded: str, image_shape: Optional[Tuple[int, int]] = None
) -> np.ndarray:
    encoded = encoded.strip()
    if encoded == "0":
        if image_shape is None:
            raise ValueError("image_shape is required when decoding an empty mask")
        return np.zeros(image_shape, dtype=np.uint8)

    if not encoded.startswith("m"):
        raise ValueError(f"Unsupported VOT mask encoding: {encoded[:32]}")

    values = [int(v.strip()) for v in encoded[1:].split(",") if v.strip()]
    if len(values) < 4:
        raise ValueError(f"Incomplete VOT mask encoding: {encoded}")

    x0, y0, width, height = values[:4]
    cropped = rle_to_mask(values[4:], width, height)

    if image_shape is None:
        return cropped

    full = np.zeros(image_shape, dtype=np.uint8)
    y1 = y0 + height
    x1 = x0 + width
    full[y0:y1, x0:x1] = cropped
    return full


def find_object_groundtruth_files(sequence_dir: Path) -> List[Path]:
    return sorted(
        p
        for p in sequence_dir.glob("groundtruth_object*.txt")
        if p.is_file() and "__ignore" not in p.name
    )


def load_initial_masks(
    sequence_dir: Path, image_shape: Tuple[int, int]
) -> Dict[int, np.ndarray]:
    object_files = find_object_groundtruth_files(sequence_dir)
    if not object_files:
        raise FileNotFoundError(
            f"No groundtruth_object*.txt files found in {sequence_dir}"
        )

    masks: Dict[int, np.ndarray] = {}
    for index, path in enumerate(object_files, start=1):
        first_line = path.read_text(encoding="utf-8").splitlines()[0].strip()
        masks[index] = decode_mask(first_line, image_shape=image_shape)
    return masks


def write_object_results(
    output_dir: Path,
    sequence_name: str,
    object_predictions: Dict[int, List[np.ndarray]],
    filename_template: str = "{sequence}_object{object_id}.txt",
) -> List[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    written_files: List[Path] = []

    for object_id, masks in sorted(object_predictions.items()):
        output_file = output_dir / filename_template.format(
            sequence=sequence_name, object_id=object_id
        )
        lines = [encode_mask(mask) for mask in masks]
        output_file.write_text("\n".join(lines) + "\n", encoding="utf-8")
        written_files.append(output_file)

    return written_files
