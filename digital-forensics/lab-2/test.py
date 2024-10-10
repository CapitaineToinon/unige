import os
import numpy as np
import matplotlib.pyplot as plt
import argparse

from shared import get_images, dataset_prnus_directory_name, cameras, get_residuals


def get_prnu(camera: str) -> np.array:
    return np.loadtxt(os.path.join(dataset_prnus_directory_name, f"{camera}.txt"))


def coeff2(K: np.ndarray, image: np.array):
    W = get_residuals(image)
    return np.corrcoef(W.flatten(), (K * image).flatten())[0, 1]


def main():
    parser = argparse.ArgumentParser(
        prog="test.py", description="Test all images against a camera"
    )

    parser.add_argument("camera", choices=cameras, default="PXL")
    args, _ = parser.parse_known_args()

    K = get_prnu(args.camera)

    offset = 0
    for camera in cameras:
        points = [coeff2(K, image) for image in get_images(f"{camera}_test")]
        plt.plot(range(offset, offset + len(points)), points, "x", label=camera)
        offset += len(points)

    plt.title(f"Testing camera {args.camera}")
    plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    plt.show()


if __name__ == "__main__":
    main()
