import argparse
import numpy as np
from path import Path
from city import City


def get_distance(item: tuple[City, float]) -> float:
    _, distance = item
    return distance


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("filename", nargs='?', type=str, default="cities3.dat")

    args, _ = parser.parse_known_args()
    path = Path.load_from_text(args.filename)
    starting_city = np.random.choice(path.cities)
    greedy_solution: list[City] = [starting_city]

    while len(greedy_solution) < len(path.cities):
        candidates = [
            (city, city.norm(greedy_solution[-1]))
            for city in path.cities
            if city not in greedy_solution
        ]

        city, _ = min(candidates, key=get_distance)
        greedy_solution.append(city)
        Path(cities=greedy_solution).show()


if __name__ == "__main__":
    main()
