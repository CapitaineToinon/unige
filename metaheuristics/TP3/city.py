import numpy as np

from dataclasses import dataclass


@dataclass
class City:
    name: str
    x: float
    y: float

    def __init__(self, name: str, x: float, y: float):
        self.name = name
        self.x = x
        self.y = y

    def coordinates(self) -> tuple[float, float]:
        return self.x, self.y

    def norm(self, other: "City") -> float:
        return np.linalg.norm(
            np.array([*self.coordinates()]) -
            np.array([*other.coordinates()])
        ).astype('float')
