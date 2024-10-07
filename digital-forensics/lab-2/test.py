import os
import numpy as np
import matplotlib.pyplot as plt
import argparse

from shared import get_images, dataset_prnus_directory_name, cameras, get_residuals


def get_prnu(camera: str) -> np.array:
    return np.loadtxt(os.path.join(dataset_prnus_directory_name, f"{camera}.txt"))


def coeff2(image: np.array):
    W = get_residuals(image)
    return np.corrcoef(W.flatten(), (K * image).flatten())[0, 1]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog="test.py",
        description="Test all images against a camera"
    )

    parser.add_argument('camera', choices=cameras)

    args = parser.parse_args()
    K = get_prnu(args.camera)

    for camera in cameras:
        points = [coeff2(image) for image in get_images(f"{camera}_test")]
        plt.plot(points, 'x', label=camera)

    plt.legend(loc='upper right')
    plt.show()
