"""
This script is only used to generate images for the documentation
Please refer the test.py, train.py for actual implementation
"""

import shared
import test
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def save_nparray_as_image(data: np.ndarray, path: str):
    Image.fromarray(np.uint8(data * 255)).save(path)

def run_test(camera: str, output: str):
    K = test.get_prnu(camera)

    plt.close()

    offset = 0
    for c in shared.cameras:
        points = [test.coeff2(K, image) for image in shared.get_images(f"{c}_test")]
        plt.plot(range(offset, offset + len(points)), points, "x", label=c)
        offset += len(points)

    plt.title(f"Testing camera {camera}")
    plt.legend(loc="upper right")
    plt.savefig(output)


def run_broken(output: str):
    K = test.get_prnu("PXL")

    plt.close()

    dirs = [
        "PXL_test_broken",
        *[f"{camera}_test" for camera in shared.cameras if camera != "PXL"]
    ]

    offset = 0
    for dir in dirs:
        points = [test.coeff2(K, image) for image in shared.get_images(dir)]
        plt.plot(range(offset, offset + len(points)), points, "x", label=dir)
        offset += len(points)

    plt.title(f"Testing camera PXL")
    plt.legend(loc="upper right")
    plt.savefig(output)

def run_fixed(output: str):
    K = test.get_prnu("PXL")

    plt.close()

    dirs = [
        "PXL_test",
        *[f"{camera}_test" for camera in shared.cameras if camera != "PXL"]
    ]

    offset = 0
    for dir in dirs:
        points = [test.coeff2(K, image) for image in shared.get_images(dir)]
        plt.plot(range(offset, offset + len(points)), points, "x", label=dir)
        offset += len(points)

    plt.title(f"Testing camera PXL")
    plt.legend(loc="upper right")
    plt.savefig(output)

def run_unknown(output: str):
    plt.close()

    offset = 0
    for camera in shared.cameras:
        K = test.get_prnu(camera)
        points = [test.coeff2(K, image) for image in shared.get_images("UNKNOWN")]
        plt.plot(range(offset, offset + len(points)), points, "x", label=camera)
        offset += len(points)

    plt.title(f"Testing unknown images against all cameras")
    plt.legend(loc="upper right")
    plt.savefig(output)

def main():
    # original = "dataset/A07_train/A07_SDR_FLAT_001.jpg"
    #
    # with Image.open(original) as img:
    #     img.save("docs/prnu_step_1.jpg")
    #
    # processed = shared.preprocess_image(original)
    # save_nparray_as_image(processed, "docs/prnu_step_2.jpg")
    #
    # residuals = shared.get_residuals(processed)
    # save_nparray_as_image(residuals, "docs/prnu_step_3.jpg")
    #
    # image_prnu = shared.compute_prnu(processed, residuals)
    # save_nparray_as_image(image_prnu, "docs/prnu_step_4.jpg")
    #
    # camera_prnu = shared.compute_prnu_for_images("A07_train")
    # save_nparray_as_image(camera_prnu, "docs/prnu_step_5.jpg")
    #
    # run_test("A07", "docs/prnu_step_6.jpg")
    # run_test("A02", "docs/prnu_step_7.jpg")
    # run_test("C01", "docs/prnu_step_8.jpg")
    #
    # run_broken("docs/prnu_step_9.jpg")
    # run_fixed("docs/prnu_step_10.jpg")
    run_unknown("docs/prnu_step_11.jpg")


if __name__ == '__main__':
    main()
