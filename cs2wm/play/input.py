"""Keyboard and mouse to 51-dim action (#28), through the same ``cs2wm.actions`` code and config as the labels.

Mouse motion is accumulated over the whole frame (DIAMOND's play keeps only the last event), converted to degrees with
``play.sensitivity * m_yaw`` and binned with the config edges. Keys ``3`` and ``4`` both map to idx 9 ("slot 3+").
Forbidden combinations (W+S, A+D, two slots at once) are resolved to a single choice.

Weapon slots, space, R and RMB are pulses, not held keys (``PULSE_KEYS``): the labels (#11) set them only on the frame
where the event happened, so a press is emitted for exactly one frame however long the key is held.

Every key and mouse button also has a "pressed during this frame" latch. At 4-6 FPS a frame lasts 170-250 ms, so a
tap shorter than that would otherwise be released before the frame ends and lost.

pygame is imported only by ``pygame_bindings``; ``InputMapper`` works on plain integers so it runs headless.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from omegaconf import DictConfig

from cs2wm.actions import ActionSpec, counts_to_deg, encode

SLOT_KEYS = ("1", "2", "3")
# Keys the labels set only on the frame of the triggering event, so play must emit them for one frame per press.
# See extract_actions.py: space <- player_jump / take-off, r <- weapon_reload, rmb <- weapon_zoom, slots <- weapon
# change. lmb (hold-filled between shots), ctrl, shift and w/a/s/d are state in the labels, so they stay held.
PULSE_KEYS = frozenset({"space", "r", "rmb", *SLOT_KEYS})
# Opposing movement keys cancel (the labels never set both: forward speed is either > +deadzone or < -deadzone).
_OPPOSITES = (("w", "s"), ("a", "d"))

# Commands returned by ``InputMapper.handle_event`` for the loop to act on.
CMD_QUIT = "quit"
CMD_RESET = "reset"
CMD_TOGGLE_OVERLAY = "toggle_overlay"


@dataclass(frozen=True)
class InputBindings:
    """Event types and key codes, so the mapper does not need pygame.

    Attributes:
        keydown: Event type of a key press.
        keyup: Event type of a key release.
        mousebuttondown: Event type of a mouse button press.
        mousebuttonup: Event type of a mouse button release.
        mousemotion: Event type of mouse motion (events carry ``rel``).
        quit: Event type of a window close.
        keys: Key code -> action key name (``w, a, s, d, space, ctrl, shift, 1, 2, 3, r``; ``4`` maps to ``3``).
        key_reset: Key code that resets the context (Backspace).
        key_overlay: Key code that toggles the overlay (F1).
        key_quit: Key code that quits (Esc).
        windowfocuslost: Event type of the window losing focus; held input is dropped so keys cannot stay stuck.
        button_lmb: Mouse button number of the left button.
        button_rmb: Mouse button number of the right button.
    """

    keydown: int
    keyup: int
    mousebuttondown: int
    mousebuttonup: int
    mousemotion: int
    quit: int
    keys: dict[int, str]
    key_reset: int
    key_overlay: int
    key_quit: int
    windowfocuslost: int = -1
    button_lmb: int = 1
    button_rmb: int = 3


def pygame_bindings() -> InputBindings:
    """Build the bindings from pygame constants (imports pygame).

    Returns:
        Bindings for real pygame events. Left and right ctrl/shift both count; ``4`` maps to slot ``3``.
    """
    import pygame as pg

    codes = {
        "w": [pg.K_w],
        "a": [pg.K_a],
        "s": [pg.K_s],
        "d": [pg.K_d],
        "space": [pg.K_SPACE],
        "ctrl": [pg.K_LCTRL, pg.K_RCTRL],
        "shift": [pg.K_LSHIFT, pg.K_RSHIFT],
        "1": [pg.K_1],
        "2": [pg.K_2],
        "3": [pg.K_3, pg.K_4],
        "r": [pg.K_r],
    }
    return InputBindings(
        keydown=pg.KEYDOWN,
        keyup=pg.KEYUP,
        mousebuttondown=pg.MOUSEBUTTONDOWN,
        mousebuttonup=pg.MOUSEBUTTONUP,
        mousemotion=pg.MOUSEMOTION,
        quit=pg.QUIT,
        keys={code: name for name, cs in codes.items() for code in cs},
        key_reset=pg.K_BACKSPACE,
        key_overlay=pg.K_F1,
        key_quit=pg.K_ESCAPE,
        windowfocuslost=pg.WINDOWFOCUSLOST,
    )


@dataclass
class InputState:
    """Input accumulated since the last frame.

    Attributes:
        held: Currently held names (action keys, plus ``lmb`` and ``rmb`` for the buttons; never slots).
        pressed: Names pressed at least once since the last frame (the tap latch).
        slot: Slot pressed since the last frame (``"1"``, ``"2"`` or ``"3"``), or None. A newer press replaces it.
        mouse_dx_counts: Accumulated horizontal mouse counts.
        mouse_dy_counts: Accumulated vertical mouse counts.
    """

    held: set[str] = field(default_factory=set)
    pressed: set[str] = field(default_factory=set)
    slot: str | None = None
    mouse_dx_counts: float = 0.0
    mouse_dy_counts: float = 0.0


class InputMapper:
    """Maps pygame events to per-frame actions.

    Args:
        cfg: A validated cs2wm config (``actions.*``, ``play.*``).
        bindings: Event types and key codes; defaults to the pygame constants.
    """

    def __init__(self, cfg: DictConfig, bindings: InputBindings | None = None) -> None:
        self.cfg = cfg
        self.spec = ActionSpec.from_config(cfg)
        self.bindings = bindings if bindings is not None else pygame_bindings()
        self.state = InputState()

    def handle_event(self, event: object) -> str | None:
        """Update the state from one pygame event (key down/up, mouse button, mouse motion).

        Args:
            event: A ``pygame.event.Event`` or a duck-typed fake in tests (``type`` plus ``key``, ``button`` or
                ``rel``).

        Returns:
            ``CMD_QUIT``, ``CMD_RESET`` or ``CMD_TOGGLE_OVERLAY`` for the control keys, else None.
        """
        b, s = self.bindings, self.state
        etype = event.type  # type: ignore[attr-defined]
        if etype == b.quit:
            return CMD_QUIT
        if etype == b.mousemotion:
            dx, dy = event.rel  # type: ignore[attr-defined]
            s.mouse_dx_counts += dx
            s.mouse_dy_counts += dy
        elif etype in (b.mousebuttondown, b.mousebuttonup):
            button = event.button  # type: ignore[attr-defined]
            name = "lmb" if button == b.button_lmb else "rmb" if button == b.button_rmb else None
            if name is not None:
                self._set(name, etype == b.mousebuttondown)
        elif etype in (b.keydown, b.keyup):
            down = etype == b.keydown
            code = event.key  # type: ignore[attr-defined]
            if down:
                if code == b.key_quit:
                    return CMD_QUIT
                if code == b.key_reset:
                    return CMD_RESET
                if code == b.key_overlay:
                    return CMD_TOGGLE_OVERLAY
            name = b.keys.get(code)
            if name in SLOT_KEYS:
                if down:
                    s.slot = name
            elif name is not None:
                self._set(name, down)
        elif etype == b.windowfocuslost:
            self.reset()
        return None

    def _set(self, name: str, down: bool) -> None:
        """Record a press or release; a press of an already-held name (OS key repeat) is not a new press."""
        s = self.state
        if down:
            if name not in s.held:
                s.pressed.add(name)
            s.held.add(name)
        else:
            s.held.discard(name)

    def frame_action(self) -> np.ndarray:
        """Return the action for the frame just finished and reset the mouse, tap and slot accumulators.

        Held names count if held at the end of the frame or pressed during it; pulse names (``PULSE_KEYS``) count
        only if pressed during it.

        Returns:
            ``float32[51]`` action built with ``cs2wm.actions.encode``.
        """
        s, play = self.state, self.cfg.play
        active = (s.held - PULSE_KEYS) | s.pressed
        for a, b in _OPPOSITES:
            if a in active and b in active:
                active -= {a, b}
        if s.slot is not None:
            active.add(s.slot)
        x_deg = counts_to_deg(s.mouse_dx_counts, float(play.sensitivity), float(play.m_yaw))
        y_deg = counts_to_deg(s.mouse_dy_counts, float(play.sensitivity), float(play.m_pitch))
        keys = active - {"lmb", "rmb"}
        action = encode(keys, "lmb" in active, "rmb" in active, x_deg, y_deg, self.spec)
        s.pressed = set()
        s.mouse_dx_counts = s.mouse_dy_counts = 0.0
        s.slot = None
        return action

    def drop_mouse(self) -> None:
        """Discard the mouse motion accumulated so far (after a stall)."""
        self.state.mouse_dx_counts = self.state.mouse_dy_counts = 0.0

    def reset(self) -> None:
        """Forget all held input (e.g. after the window loses focus)."""
        self.state = InputState()
