import string

import numpy as np
import matplotlib.pyplot as plt

import city
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
            ('city', 'U32'),
            ('x', float),
            ('y', float)
        ])

        return Path(cities=[City(*city) for city in data])

    @staticmethod
    def generate_circular_path(n_cities: int, radius=1) -> "Path":
        if n_cities > len(string.ascii_lowercase):
            raise ValueError("n_cities must be <= 26")

        angles = np.linspace(0.0, 2 * np.pi, n_cities, endpoint=False)
        points = np.column_stack((radius * np.cos(angles), radius * np.sin(angles)))

        cities = [
            City(str(letter), *points)
            for letter, points in zip(string.ascii_lowercase, points)
        ]

        return Path(cities=cities)

    @staticmethod
    def travel(path: "Path"):
        total = len(path.cities)
        for i in range(total):
            yield path.cities[i], path.cities[(i + 1) % total]

    def __init__(self, *, cities: list[City]):
        self.cities = cities

    def copy(self) -> "Path":
        return Path(cities=self.cities.copy())

    def fitness_value(self) -> float:
        return np.sum([
            a.norm(b)
            for a, b in Path.travel(self)
        ]).astype("float")

    def show(self):
        for a, b in Path.travel(self):
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

    def save_as_text(self, filename: str):
        points = np.array([
            (city.name, city.x, city.y)
            for city in self.cities
        ], dtype=[
            ('city', 'U32'),
            ('x', float),
            ('y', float)
        ])

        np.savetxt(filename, points, fmt="%s %f %f")
