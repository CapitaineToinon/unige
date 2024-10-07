import os
import numpy as np
from shared import dataset_prnus_directory_name, compute_prnu_for_images, cameras


def main() -> None:
    os.makedirs(dataset_prnus_directory_name, exist_ok=True)

    for camera in cameras:
        prnu = compute_prnu_for_images(f"{camera}_train")
        np.savetxt(os.path.join(dataset_prnus_directory_name, f"{camera}.txt"), prnu)


if __name__ == "__main__":
    main()
