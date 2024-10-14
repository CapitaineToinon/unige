import argparse

from path import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("n_cities", nargs='?', type=int, default=10)
    parser.add_argument("--output", type=str, default="cities3.dat")

    args, _ = parser.parse_known_args()
    Path.generate_circular_path(args.n_cities).save_as_text(args.output)


if __name__ == "__main__":
    main()
