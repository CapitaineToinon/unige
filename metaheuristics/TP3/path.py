from functools import lru_cache

import numpy as np
import matplotlib.pyplot as plt

from city import City
from dataclasses import dataclass


def random_ij(n: int) -> tuple[int, int]:
    while True:
        i, j = np.random.randint(0, n, 2)
        if i == j:
            continue
        return i, j


@dataclass
class Path:
    cities: list[City]

    @staticmethod
    def load_from_text(path: str) -> "Path":
        data = np.loadtxt(path, dtype=[
            ('city', str),
            ('x', float),
            ('y', float)
        ])

        return Path(cities=[City(*city) for city in data])

    @staticmethod
    def generate_circular_path(n_cities: int, radius=1) -> "Path":
        angles = np.linspace(0.0, 2 * np.pi, n_cities, endpoint=False)
        points = np.column_stack((radius * np.cos(angles), radius * np.sin(angles)))

        digit_count = np.floor(1 + np.log10(n_cities)).astype("int")
        city_name_format = f"c{{:0{digit_count}d}}"

        cities = [
            City(city_name_format.format(index), *points)
            for index, points in enumerate(points)
        ]

        return Path(cities=cities)

    def __init__(self, *, cities: list[City]):
        self.cities = cities

    def __hash__(self):
        return hash(";".join([city.name for city in self.cities]))

    def __len__(self) -> int:
        return len(self.cities)

    def __sub__(self, other: "Path") -> float:
        return self.fitness_value() - other.fitness_value()

    def __lt__(self, other: "Path") -> bool:
        return self.fitness_value() < other.fitness_value()

    def __gt__(self, other: "Path") -> bool:
        return self.fitness_value() > other.fitness_value()

    def __le__(self, other: "Path") -> bool:
        return self.fitness_value() <= other.fitness_value()

    def __ge__(self, other: "Path") -> bool:
        return self.fitness_value() >= other.fitness_value()

    def copy(self) -> "Path":
        return Path(cities=self.cities.copy())

    def travel(self):
        total = len(self.cities)
        for i in range(total):
            yield self.cities[i], self.cities[(i + 1) % total]

    @lru_cache(maxsize=None)
    def fitness_value(self) -> float:
        return np.sum([
            a.norm(b)
            for a, b in self.travel()
        ]).astype("float")

    def show(self):
        for a, b in self.travel():
            plt.plot([a.x, b.x], [a.y, b.y], "bo-")

        plt.title(f"f(x) = {self.fitness_value()}")
        plt.gca().set_aspect('equal')
        plt.show()

    def random_swap(self) -> "Path":
        cities = self.cities.copy()
        i, j = random_ij(len(cities))
        cities[i], cities[j] = cities[j], cities[i]
        return Path(cities=cities)

    def shuffle(self) -> "Path":
        return Path(cities=np.random.permutation(self.cities).tolist())

    def as_points(self):
        return np.array([
            (city.name, city.x, city.y)
            for city in self.cities
        ], dtype=[
            ('city', 'U32'),
            ('x', float),
            ('y', float)
        ])

    def save_as_text(self, filename: str):
        np.savetxt(filename, self.as_points(), fmt="%s %f %f")
