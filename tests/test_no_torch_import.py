"""The Windows night pipeline must import and run without torch (Smart App Control blocks its DLLs, #87)."""

import subprocess
import sys
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

CODE = textwrap.dedent(
    """
    import sys

    sys.modules["torch"] = None  # makes `import torch` raise ImportError
    sys.modules["torchvision"] = None

    import numpy as np

    import cs2wm.actions
    import cs2wm.cli
    import cs2wm.config
    import cs2wm.demo
    import cs2wm.eval.label_accuracy
    import cs2wm.eval.render_offset
    import cs2wm.eval.rotation
    import cs2wm.pipeline.pack
    import cs2wm.pipeline.supervisor
    import cs2wm.timeline

    frames = np.random.default_rng(0).integers(0, 256, (3, 150, 280, 3), dtype=np.uint8)
    out = cs2wm.pipeline.pack.to_low_res(frames)
    assert out.shape == (3, 30, 56, 3) and out.dtype == np.uint8
    print("ok")
    """
)


def test_pipeline_imports_and_resizes_without_torch() -> None:
    res = subprocess.run(
        [sys.executable, "-c", CODE], check=False, cwd=ROOT, capture_output=True, text=True, timeout=300
    )
    assert res.returncode == 0, res.stderr
    assert res.stdout.strip().endswith("ok")
