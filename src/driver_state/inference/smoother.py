from __future__ import annotations

from collections import deque
import numpy as np


class ProbabilitySmoother:
    def __init__(self, window: int):
        self.window = int(window)
        self.buffer = deque(maxlen=self.window)

    def update(self, probs: np.ndarray) -> np.ndarray:
        self.buffer.append(np.asarray(probs, dtype="float32"))
        return np.mean(np.stack(self.buffer), axis=0)

    def reset(self) -> None:
        self.buffer.clear()
