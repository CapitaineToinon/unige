import numpy as np
from path import Path


def compute_initial_temperature(path: Path, *, n=100) -> float:
    current_path = path
    mean_energy = 0.0

    for _ in range(n):
        new_path = current_path.random_swap()
        new_energy = np.abs(new_path.fitness_value() - current_path.fitness_value())
        mean_energy += new_energy * 1 / n
        current_path = new_path

    return mean_energy / -np.log(0.5)


def compute_P(*, current_path: Path, next_path: Path, temperature: float) -> float:
    delta = next_path.fitness_value() - current_path.fitness_value()

    if delta < 0:
        return 1.0

    return min(1.0, np.exp(-delta / temperature))


def solve_until_freeze(initial_path: Path, *, stop_after_no_improvements=3, stop_after_accepted=12,
                       stop_after_total=100) -> Path:
    current_path = initial_path.copy()
    best_path = current_path.copy()

    temperature = compute_initial_temperature(current_path)
    no_improvements = 0

    while no_improvements < stop_after_no_improvements:
        accepted = 0
        total = 0
        current_improved = False

        while accepted < stop_after_accepted and total < stop_after_total:
            next_path = current_path.random_swap()
            probability = compute_P(current_path=current_path, next_path=next_path, temperature=temperature)

            if np.random.random() < probability:
                accepted += 1

                if next_path.fitness_value() < best_path.fitness_value():
                    best_path = next_path
                    best_path.show()

                if next_path.fitness_value() < current_path.fitness_value():
                    current_improved = True

                current_path = next_path

            total += 1

        if not current_improved:
            no_improvements += 1

        temperature *= 0.9

    return best_path


def main():
    initial_path = Path.load_from_text("cities3.dat")
    current_path = initial_path.shuffle()
    best_path = solve_until_freeze(current_path)
    best_path.show()


if __name__ == "__main__":
    main()
