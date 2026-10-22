"""Keyboard and mouse to 51-dim action (#28), through the same ``cs2wm.actions`` code and config as the labels.

Mouse motion is accumulated over the whole frame (DIAMOND's play keeps only the last event), converted to degrees with
``play.sensitivity * m_yaw`` and binned with the config edges. Keys ``3`` and ``4`` both map to idx 9 ("slot 3+").
Forbidden combinations (W+S, A+D, two slots at once) are resolved to a single choice.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from omegaconf import DictConfig


@dataclass
class InputState:
    """Input accumulated since the last frame.

    Attributes:
        keys: Currently pressed key names.
        lmb: Left button held.
        rmb: Right button held.
        mouse_dx_counts: Accumulated horizontal mouse counts.
        mouse_dy_counts: Accumulated vertical mouse counts.
    """

    keys: set[str] = field(default_factory=set)
    lmb: bool = False
    rmb: bool = False
    mouse_dx_counts: float = 0.0
    mouse_dy_counts: float = 0.0


class InputMapper:
    """Maps pygame events to per-frame actions.

    Args:
        cfg: A validated cs2wm config (``actions.*``, ``play.*``).
    """

    def __init__(self, cfg: DictConfig) -> None:
        self.cfg = cfg
        self.state = InputState()

    def handle_event(self, event: object) -> None:
        """Update the state from one pygame event (key down/up, mouse button, mouse motion).

        Args:
            event: A ``pygame.event.Event`` or a duck-typed fake in tests.
        """
        # TODO(#28): KEYDOWN/KEYUP, MOUSEBUTTONDOWN/UP, MOUSEMOTION (accumulate rel).
        raise NotImplementedError("TODO(#28): InputMapper.handle_event")

    def frame_action(self) -> np.ndarray:
        """Return the action for the frame just finished and reset the mouse accumulators.

        Returns:
            ``float32[51]`` action built with ``cs2wm.actions.encode``.
        """
        # TODO(#28): counts_to_deg(sensitivity, m_yaw), then encode.
        raise NotImplementedError("TODO(#28): InputMapper.frame_action")
