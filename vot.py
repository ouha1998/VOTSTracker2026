from __future__ import annotations

import collections
import os
from typing import Iterable

import numpy as np

_USE_TRAX = os.environ.get("VOT_USE_TRAX", "1") == "1"

try:
    import trax

    if trax._ctypes.trax_version().decode("ascii") < "4.0.0" and _USE_TRAX:
        raise ImportError("TraX version 4.0.0 or newer is required.")
except ImportError:
    _USE_TRAX = False

Rectangle = collections.namedtuple("Rectangle", ["x", "y", "width", "height"])
Point = collections.namedtuple("Point", ["x", "y"])
Polygon = collections.namedtuple("Polygon", ["points"])
Mask = collections.namedtuple("Mask", ["mask", "offset"])
Empty = collections.namedtuple("Empty", [])


def _parse_region(line: str):
    line = line.strip()
    if line[0] == "m":
        encoded = [int(x) for x in line[1:].split(",")]
        ox, oy, width, height = encoded[:4]
        v = np.zeros(width * height, dtype=np.uint8)
        idx_ = 0
        for i in range(len(encoded[4:])):
            if i % 2 != 0:
                for j in range(encoded[4 + i]):
                    v[idx_ + j] = 1
            idx_ += encoded[4 + i]
        v = v.reshape((height, width))
        m_ = np.zeros((oy + height, ox + width), dtype=np.uint8)
        m_[oy : oy + height, ox : ox + width] = v
        return m_

    tokens = [float(t) for t in line.split(",")]
    if len(tokens) == 1:
        return Empty()
    if len(tokens) == 2:
        return Point(tokens[0], tokens[1])
    if len(tokens) == 4:
        return Rectangle(tokens[0], tokens[1], tokens[2], tokens[3])
    if len(tokens) % 2 == 0 and len(tokens) > 4:
        return Polygon([Point(x_, y_) for x_, y_ in zip(tokens[::2], tokens[1::2])])
    return None


def _encode_region(region) -> str:
    if isinstance(region, Empty):
        return "0"
    if isinstance(region, Point):
        return f"{region.x},{region.y}"
    if isinstance(region, Rectangle):
        return f"{region.x},{region.y},{region.width},{region.height}"
    if isinstance(region, Polygon):
        return ",".join(f"{point.x},{point.y}" for point in region.points)
    if isinstance(region, Mask):
        return _encode_region(region.mask)
    if isinstance(region, np.ndarray):
        ys, xs = np.nonzero(region)
        if len(xs) == 0 or len(ys) == 0:
            return "0"
        ox, oy = xs.min(), ys.min()
        width, height = xs.max() - ox + 1, ys.max() - oy + 1
        v = region[oy : oy + height, ox : ox + width].flatten()
        # VOT mask RLE always starts with the background run length, even if it is 0.
        rle = []
        current = 0
        count = 0
        for value in v:
            value = 1 if value else 0
            if value == current:
                count += 1
            else:
                rle.append(count)
                current = value
                count = 1
        rle.append(count)
        return f"m{ox},{oy},{width},{height}," + ",".join(str(x) for x in rle)
    raise TypeError(f"Unsupported region type: {type(region)!r}")


def _validate_region(region, valid_formats) -> bool:
    if isinstance(region, Empty):
        return "empty" in valid_formats
    if isinstance(region, Point):
        return "point" in valid_formats
    if isinstance(region, Rectangle):
        return "rectangle" in valid_formats
    if isinstance(region, Polygon):
        return "polygon" in valid_formats
    if isinstance(region, (Mask, np.ndarray)):
        return "mask" in valid_formats
    return False


class VOT:
    """Official-style VOT Python wrapper with multi-object support."""

    def __init__(self, region_format, channels=None, multiobject: bool | None = None):
        if _USE_TRAX:
            assert region_format in ["rectangle", "polygon", "mask"]
        else:
            assert region_format in ["rectangle", "polygon", "mask", "point"]

        if multiobject is None:
            multiobject = os.environ.get("VOT_MULTI_OBJECT", "0") == "1"

        if channels is None:
            channels = ["color"]
        elif channels == "rgbd":
            channels = ["color", "depth"]
        elif channels == "rgbt":
            channels = ["color", "ir"]
        elif channels == "ir":
            channels = ["ir"]
        else:
            raise RuntimeError(f"Illegal configuration {channels}.")

        self._trax = None
        self._multiobject = multiobject

        if _USE_TRAX:
            self._trax = trax.Server(
                [region_format],
                ["path"],
                channels,
                metadata=dict(vot="python"),
                multiobject=multiobject,
            )
            request = self._trax.wait()
            assert request.type == "initialize"
            self._objects = []
            assert len(request.objects) > 0 and (multiobject or len(request.objects) == 1)
            for tobject, _ in request.objects:
                if isinstance(tobject, trax.Polygon):
                    self._objects.append(Polygon([Point(x[0], x[1]) for x in tobject]))
                elif isinstance(tobject, trax.Mask):
                    mask_array = tobject.array(True)
                    offset = (0, 0)
                    if hasattr(tobject, "offset"):
                        raw_offset = tobject.offset
                        if callable(raw_offset):
                            raw_offset = raw_offset()
                        try:
                            offset = (int(raw_offset[0]), int(raw_offset[1]))
                        except Exception:
                            offset = (0, 0)
                    elif hasattr(tobject, "bounds"):
                        try:
                            bounds = tobject.bounds()
                            offset = (int(bounds[0]), int(bounds[1]))
                        except Exception:
                            offset = (0, 0)
                    self._objects.append(Mask(mask_array, offset))
                else:
                    self._objects.append(Rectangle(*tobject.bounds()))
            self._image = [x.path() for _, x in request.image.items()]
            if len(self._image) == 1:
                self._image = self._image[0]
            self._trax.status(request.objects)
        else:
            self._frames = []
            self._objects = []
            self._object_keys = []
            self._object_trajectory = []
            frames = []
            for channel in channels:
                filename = f"frames_{channel}.txt"
                if not os.path.exists(filename):
                    raise RuntimeError(f"Missing frames file for channel {channel}")
                with open(filename, "r", encoding="utf-8") as f:
                    frames.append([line.strip() for line in f])

            if len(frames) == 1:
                self._frames = frames[0]
            else:
                assert all(len(f) == len(frames[0]) for f in frames)
                self._frames = list(zip(*frames))

            queries = [f for f in os.listdir() if f.startswith("query_") and f.endswith(".txt")]
            assert len(queries) > 0, "No query file found"

            for query_file in queries:
                with open(query_file, "r", encoding="utf-8") as f:
                    object_id = query_file[len("query_") : -len(".txt")]
                    lines = [line.strip() for line in f if line.strip()]
                    offset = int(lines[0])
                    if offset != 0:
                        raise RuntimeError(
                            f"Only offset 0 is supported in the wrapper, but got offset {offset} in file {query_file}"
                        )
                    state = _parse_region(lines[1])
                    if not _validate_region(state, [region_format]):
                        raise RuntimeError(f"Invalid region format in file {query_file}")
                    self._objects.append(state)
                    self._object_keys.append(object_id)
                    self._object_trajectory.append([state])
            self._position = 0

    def region(self):
        assert not self._multiobject
        return self.objects()[0]

    def objects(self):
        return self._objects

    def report(self, status):
        def convert_trax(region):
            if region is None or isinstance(region, Empty):
                try:
                    return trax.Special(0)
                except Exception:
                    return trax.Rectangle.create(0, 0, 0, 0)
            if isinstance(region, Polygon):
                return trax.Polygon.create([(x.x, x.y) for x in region.points])
            if isinstance(region, Mask):
                return trax.Mask.create(region.mask)
            if isinstance(region, np.ndarray):
                return trax.Mask.create(region)
            if isinstance(region, Rectangle):
                return trax.Rectangle.create(region.x, region.y, region.width, region.height)
            raise TypeError(f"Unsupported region type for reporting: {type(region)!r}")

        if not _USE_TRAX:
            if not self._multiobject:
                status = [status]
            assert len(self._object_keys) == len(status), "Number of status entries must match the number of objects"
            for key, state in enumerate(status):
                self._object_trajectory[key].append(state)
            return

        if not self._multiobject:
            status = [convert_trax(status)]
        else:
            assert isinstance(status, (list, tuple))
            assert len(status) == len(self._objects), "Output object count must match initialization object count"
            status = [(convert_trax(x), {}) for x in status]
        self._trax.status(status, {})

    def frame(self):
        if not _USE_TRAX:
            if self._position >= len(self._frames):
                return None
            frame = self._frames[self._position]
            self._position += 1
            return frame

        if hasattr(self, "_image"):
            image = self._image
            del self._image
            return image

        request = self._trax.wait()
        assert request.objects is None or len(request.objects) == 0
        if request.type == "frame":
            image = [x.path() for _, x in request.image.items()]
            if len(image) == 1:
                return image[0]
            return image
        return None

    def quit(self):
        if _USE_TRAX and hasattr(self, "_trax"):
            self._trax.quit()
        if not _USE_TRAX:
            for key, trajectory in zip(self._object_keys, self._object_trajectory):
                with open(f"output_{key}.txt", "w", encoding="utf-8") as f:
                    for state in trajectory:
                        f.write(_encode_region(state) + "\n")
            self._object_trajectory = []

    def __del__(self):
        self.quit()


class VOTManager:
    def __init__(self, factory, region_format, channels=None):
        self._handle = VOT(region_format, channels, multiobject=True)
        self._factory = factory

    def run(self):
        objects = self._handle.objects()
        image = self._handle.frame()
        if not image:
            return
        trackers = [self._factory(image, obj) for obj in objects]

        while True:
            image = self._handle.frame()
            if not image:
                break
            status = [tracker(image) for tracker in trackers]
            self._handle.report(status)
        self._handle.quit()
