# Adapted from DIAMOND (Alonso et al., NeurIPS 2024), https://github.com/eloialonso/diamond (MIT License),
# via diamond-cs2 v1 (https://github.com/cooperwooley/diamond-cs2 @ a0b9e24). See ATTRIBUTION.md.
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, Union

import torch


@dataclass
class SegmentId:
    episode_id: Union[int, str]
    start: int
    stop: int


@dataclass
class Segment:
    obs: torch.FloatTensor
    act: torch.LongTensor
    rew: torch.FloatTensor
    end: torch.ByteTensor
    trunc: torch.ByteTensor
    mask_padding: torch.BoolTensor
    info: Dict[str, Any]
    id: SegmentId

    @property
    def effective_size(self):
        return self.mask_padding.sum().item()
