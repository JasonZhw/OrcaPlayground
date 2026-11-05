"""Preview keeps all three camera views at their recorded pixel dimensions."""

import json
import sys
from fractions import Fraction

import av
import numpy as np
import pytest

from examples.euler.medication_sorting.preview import main
from examples.euler.medication_sorting.recording import CAMERAS


@pytest.mark.parametrize("speed", [1, 2])
def test_preview_preserves_native_camera_sizes(tmp_path, monkeypatch, speed):
    cameras = {}
    for index, name in enumerate(CAMERAS):
        path = tmp_path / f"{name}.mkv"
        with av.open(str(path), "w") as output:
            stream = output.add_stream("ffv1", rate=50)
            stream.width, stream.height, stream.pix_fmt = 64, 48, "bgr0"
            for i in range(4):
                frame = av.VideoFrame.from_ndarray(np.full((48, 64, 3), 60 + index * 50, np.uint8), format="rgb24")
                frame.pts, frame.time_base = i, Fraction(1, 50)
                for packet in stream.encode(frame):
                    output.mux(packet)
            for packet in stream.encode():
                output.mux(packet)
        cameras[name] = {"path": path.name, "native_shape": [48, 64, 3]}
    (tmp_path / "metadata.json").write_text(json.dumps({"raw_valid": True, "fps": 50, "cameras": cameras}))
    target = tmp_path / "preview.mp4"
    monkeypatch.setattr(
        sys, "argv", ["preview", "--episode", str(tmp_path), "--output", str(target), "--speed", str(speed)]
    )
    main()
    with av.open(str(target)) as video:
        frames = list(video.decode(video=0))
        assert video.streams.video[0].average_rate == 50 * speed
    assert len(frames) == 4
    assert frames[0].width == 3 * 64
    assert frames[0].height == 48 + 32
