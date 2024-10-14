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

    def __hash__(self):
        return hash(self.name)

    def coordinates(self) -> np.ndarray:
        return np.array([self.x, self.y])

    def norm(self, other: "City") -> float:
        return np.linalg.norm(self.coordinates() - other.coordinates()).astype('float')
