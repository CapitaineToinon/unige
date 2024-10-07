import os
from functools import reduce

from skimage import restoration
import numpy as np
from PIL import Image

dataset_directory_name = "dataset"
dataset_prnus_directory_name = "prnu"
cameras = ["A02", "A07", "I03"]


def show_image(data: np.array) -> None:
    Image.fromarray(np.uint8(data * 255)).show()


def get_images(folder: str) -> list[np.array]:
    filenames = os.listdir(os.path.join(dataset_directory_name, folder))

    paths = [
        os.path.join(dataset_directory_name, folder, filename)
        for filename in filenames
    ]

    paths = filter(lambda path: path.endswith(".jpg"), paths)

    for path in paths:
        with Image.open(path) as img:
            image = img.crop((0, 0, 1024, 1024))
            image = image.convert('L')
            yield np.array(image).astype('float64') / 255


def compute_prnu_for_images(folder: str):
    prnus = [denoise_and_prnu(img) for img in get_images(folder)]
    return reduce(lambda a, b: a + b, prnus) / len(prnus)


def get_residuals(image: np.array) -> np.array:
    denoised = restoration.denoise_wavelet(image)
    return image - denoised


def denoise_and_prnu(image: np.array) -> np.array:
    residuals = get_residuals(image)
    return compute_prnu(image, residuals)


def compute_prnu(I: np.array, W: np.array) -> np.array:
    return (W * I) / (I ** 2)
