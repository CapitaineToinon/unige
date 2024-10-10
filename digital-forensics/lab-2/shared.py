import os
from functools import reduce

from skimage import restoration
import numpy as np
from PIL import Image

dataset_directory_name = "dataset"
dataset_prnus_directory_name = "prnu"
cameras = ["A02", "A07", "I03", "C01", "PXL"]


def show_image(data: np.ndarray) -> None:
    Image.fromarray(np.uint8(data * 255)).show()

def preprocess_image(path: str) -> np.ndarray:
    with Image.open(path) as img:
        image = img.crop((0, 0, 1024, 1024))
        image = image.convert("L")
        return np.array(image).astype("float64") / 255

def get_images(folder: str) -> list[np.ndarray]:
    filenames = os.listdir(os.path.join(dataset_directory_name, folder))

    paths = [
        os.path.join(dataset_directory_name, folder, filename) for filename in filenames
    ]

    paths = [path for path in paths if path.endswith(".jpg")]

    for path in paths:
        yield preprocess_image(path)


def compute_prnu_for_images(folder: str):
    prnus = [denoise_and_prnu(img) for img in get_images(folder)]
    return reduce(lambda a, b: a + b, prnus) / len(prnus)

def get_denoised_image(image: np.ndarray) -> np.ndarray:
    return restoration.denoise_wavelet(image)

def get_residuals(image: np.ndarray) -> np.ndarray:
    denoised = get_denoised_image(image)
    return image - denoised

def denoise_and_prnu(image: np.ndarray) -> np.ndarray:
    residuals = get_residuals(image)
    return compute_prnu(image, residuals)


def compute_prnu(I: np.ndarray, W: np.ndarray) -> np.ndarray:
    a = W * I
    b = I ** 2
    return np.divide(a, b, out=np.zeros_like(a), where=b != 0)
