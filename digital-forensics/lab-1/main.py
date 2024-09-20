import os
import numpy as np
from skimage import metrics
from PIL import Image, ImageFilter

images = [
    ("pexels-icsa.jpg", 256),
    ("pexels-jibarofoto.jpg", 512),
    ("pexels-lecreusois.jpg", 1000),
    ("pexels-olly.jpg", 2000),
]

image_folder = 'images'
resized_folder = 'resized_images'
distorted_folder = 'distorted_images'

os.makedirs(resized_folder, exist_ok=True)
os.makedirs(distorted_folder, exist_ok=True)


def prepare_images(sources: list[tuple[str, int]]) -> list[tuple[Image, Image]]:
    return [prepare_image(image) for image in sources]


def prepare_image(image: tuple[str, int]) -> tuple[Image, Image]:
    filename, size = image
    file_path = os.path.join(image_folder, filename)

    with Image.open(file_path) as img:
        resized_filepath = os.path.join(resized_folder, filename)
        img.thumbnail((size, size))
        img = img.convert('L')
        img.save(resized_filepath)

        # Now distort the image
        distorted_filepath = os.path.join(distorted_folder, filename)
        distorted = img.filter(ImageFilter.GaussianBlur(radius=10))
        distorted.save(distorted_filepath, format="JPEG", quality=80, optimize=True)

        return np.array(img), np.array(distorted)


def main():
    image_data = prepare_images(images)

    for i in range(0, len(image_data)):
        a, b = image_data[i]
        print(f"--- {images[i][0]} ")
        print(metrics.mean_squared_error(a, b))
        print(metrics.peak_signal_noise_ratio(a, b))
        print(metrics.structural_similarity(a, b))


if __name__ == '__main__':
    main()
