import os
import numpy as np
import itertools
from PIL import Image, ImageFilter

# https://www.scirp.org/journal/paperinformation?paperid=90911

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
        img.save(resized_filepath)

        # Now distort the image
        distorted_filepath = os.path.join(distorted_folder, filename)
        distorted = img.filter(ImageFilter.GaussianBlur(radius=2))
        distorted.save(distorted_filepath, format="JPEG", quality=80, optimize=True)

        return img, distorted


def mse(image_a: Image, image_b: Image) -> float:
    assert image_a.size == image_b.size

    np.mean(image_a)

    width, height = image_a.size

    return np.sum([
        np.sum([(rgb_b - rgb_a) ** 2 for rgb_a, rgb_b in zip(image_a.getpixel(coords), image_b.getpixel(coords))])
        for coords in itertools.product(range(0, width), range(0, height))
    ]) / (width * height * 3)

def rmse(image_a: Image, image_b: Image) -> float:
    return np.sqrt(mse(image_a, image_b))


def main():
    image_data = prepare_images(images)

    for data in image_data:
        a, b = data
        print(mse(a, b))
        print(rmse(a, b))
        break


if __name__ == '__main__':
    main()
