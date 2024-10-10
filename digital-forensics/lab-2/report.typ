#import "@local/unige:0.1.0": *

#show: doc => conf(
  title: [
    Photo Response Non Uniformity (PRNU) estimation
  ],
  header: "Université de Genève",
  subheader: [
    Digital Forensics\
    14X065
  ],
  authors: (
    (
      name: "Antoine Sutter",
      email: "antoine.sutter@etu.unige.ch",
    ),
  ),
  doc,
)

= Dataset

As suggested in the instructions, I used the sample dataset provided on moodle. Each camera has a `_train` folder containing flat pictures used to compute the PRNU and a `_test` folder containing normal images taken by the same camera used to test the correlations against the PRNU.

```
dataset
├── A02_test
├── A02_train
├── A07_test
├── A07_train
├── I03_test
└── I03_train
```

Once we have everything working with this dataset, I will add new cameras.

#pagebreak()

= Compute the PRNU

To compute the PRNU of a camera, a few steps are required.

== Preprocess the images <preprocess>

For each image of the dataset, the image is preprocessed to be standardized. This includes converting the image to gray scale and cropping the image to a square of $1024 times 1024$. Then, the pixel values are also normalized to be floats between 0 and 1 and convert it to a numpy #footnote[https://numpy.org/] array. Image manipulation is done using the Pillow #footnote[https://python-pillow.org/] library.

```python
import numpy as np
from PIL import Image

with Image.open(path) as img:
    image = img.crop((0, 0, 1024, 1024))
    image = image.convert("L")
    yield np.array(image).astype("float64") / 255
```

We also use a python generator to be able process one image at a time, preventing to load all images at once in memory.

#grid(
  columns: (1fr, 1fr),
  gutter: 24pt,
  align: center + horizon,
  figure(
    image("docs/prnu_step_1.jpg"),
    caption: [Original image],
  ),
  figure(
    image("docs/prnu_step_2.jpg"),
    caption: [$1024 times 1024$ black and white],
  ),
)

#pagebreak()

== Denoising and residuals

One the image is preprocessed, the next step is to denoise the image. The instructions suggest using the Wienner2 filter but for ease of use, I ended up using wavelet filter through the scikit-image #footnote[https://scikit-image.org/] library. The residual $W_I$ is defined the difference between the original image $I$ and the denoised image $F(I)$:

$ W_I = I - F(I) $

In python, this is done as such:

```python
import numpy as np
from skimage import restoration

def get_denoised_image(image: np.ndarray) -> np.ndarray:
    return restoration.denoise_wavelet(image)

def get_residuals(image: np.ndarray) -> np.ndarray:
    denoised = get_denoised_image(image)
    return image - denoised
```

#grid(
  columns: (1fr, 1fr),
  gutter: 24pt,
  align: center + horizon,
  figure(
    image("docs/prnu_step_2.jpg"),
    caption: [Preprocessed image],
  ),
  figure(
    image("docs/prnu_step_3.jpg"),
    caption: [Residuals],
  ),
)

#pagebreak()

== Image PRNU

The PRNU is defined as such for image $i$:

$ (W^i I^i) / (I^i)^2 $

In python we can do it as such:

```python
import numpy as np

def compute_prnu(I: np.ndarray, W: np.ndarray) -> np.ndarray:
    return (W * I) / (I ** 2)
```

However, because an image can contain completely black pixels with the value 0 which will lead to `nan` (not a number) values. To avoid this, we can use numpy's `divide` function to just return 0 when python is trying to do a division by 0.

```python
import numpy as np

def compute_prnu(I: np.ndarray, W: np.ndarray) -> np.ndarray:
    a = W * I
    b = I ** 2
    return np.divide(a, b, out=np.zeros_like(a), where=b != 0)
```

#grid(
  columns: (1fr, 1fr),
  gutter: 24pt,
  align: center + horizon,
  figure(
    image("docs/prnu_step_2.jpg"),
    caption: [Preprocessed image],
  ),
  figure(
    image("docs/prnu_step_4.jpg"),
    caption: [Image's PRNU],
  ),
)

This wasn't a problem with the same data provided in moodle, however when I imported by own images, some of them had completely black pixels so I had to write to workaround.

#pagebreak()

== Camera PRNU

Finally, we repeat the same process for all images in the training dataset. The PRNU for the camera is defined as the sum of all PRNU over the amount of images:

$ K = (sum_(i=1)^N (W^i I^i)) / (sum_(i=1)^N (I^i)^2) $

In python, it is done as such:

```python
def compute_prnu_for_images(folder: str):
    prnus = [denoise_and_prnu(img) for img in get_images(folder)]
    return reduce(lambda a, b: a + b, prnus) / len(prnus)
```

And if we try to visualize the final PRNU, it looks something like this:

#figure(
  image("docs/prnu_step_5.jpg", width: 80%),
  caption: [Camera's PRNU],
)

#pagebreak()

= Test images

To avoid recomputing the PRNU's of each camera over and over again, the PRNU matrices are saved in a `prnu` directory.

```
prnu
├── A02.txt
├── A07.txt
├── C01.txt
├── I03.txt
└── PXL.txt
```

We can then load back the PRNU using numpy:

```python
import os
import numpy as np

def get_prnu(camera: str) -> np.array:
    return np.loadtxt(os.path.join("prnu", f"{camera}.txt"))
```

We can now test images and plot their correlations with each camera's PRNU. Each test image has to also be preprocessed the same way training images were, like explained in @preprocess. Then, computed the residual matrix for the test image and compute correlation coefficients between the residual matrix and the camera's PRNU.

To compate image $I$ with image $J$, the coefficient is defined as:

$ p = "corr"(W_J, K J) $

Where $W_J$ is the residual matrix of $J$ and $K$ is the PRNU of the camera currently being tested against. In python, we can do it as such:

```py
import numpy as np

def coeff2(K: np.ndarray, image: np.array):
    W = get_residuals(image)
    return np.corrcoef(W.flatten(), (K * image).flatten())[0, 1]
```

We can now loop on all test images and test against a given camera that is passed as an argument to the script.

```py
K = get_prnu(args.camera)

offset = 0
for camera in cameras:
    points = [coeff2(K, image) for image in get_images(f"{camera}_test")]
    plt.plot(range(offset, offset + len(points)), points, "x", label=camera)
    offset += len(points)

plt.title(f"Testing camera {args.camera}")
plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
plt.show()
```

#pagebreak()

== Testing against a camera

If we run the test script against the A07 camera, here is the result we have:

#figure(
  image("docs/prnu_step_6.jpg", width: 75%),
  caption: [A07 test results],
)

Because we know which camera too each test picture, we can color-code them accordingly and verify that that only the correct images have a high correlation value. In this case, we're running the images against the camera A07 and we can see that we correctly only get the A07 test images to have a high correlation. We can therefore be confident that the algorythm works correctly. Here is the results when testing against camera A02 instead.

#figure(
  image("docs/prnu_step_7.jpg", width: 75%),
  caption: [A02 test results],
)

#pagebreak()

== Adding the C01 camera

Before adding my Google Pixel 5 as a test camera, I searched for a dataset online I could use and found one at #link("https://lesc.dinfo.unifi.it/materials/datasets/") #footnote[https://lesc.dinfo.unifi.it/materials/datasets/]. I only imported a single camera, the C01 and ran the tests against that new camera.

#figure(
  image("docs/prnu_step_8.jpg", width: 80%),
  caption: [C01 test results],
)

#pagebreak()

== Adding the Google Pixel 5 camera

Finally, the ultimate test was to add my own dataset. I used my smartphone, a Google Pixel 5 and took a few pictures of flat surfaces.

#pad(y: 20pt)[
  #grid(
    columns: (1fr, 1fr),
    gutter: 24pt,
    align: center + horizon,
    figure(
      image("dataset/PXL_train/PXL_20241010_141921916.jpg"),
      caption: [PXL_20241010_141921916],
    ),
    figure(
      image("dataset/PXL_train/PXL_20241010_141938337.jpg"),
      caption: [PXL_20241010_141938337],
    ),
  )
]

And then normal photos for the test data.

#pad(y: 20pt)[
  #grid(
    columns: (1fr, 1fr),
    gutter: 24pt,
    align: center + horizon,
    figure(
      image("dataset/PXL_test_broken/PXL_20241010_142028717.jpg", width: 50%),
      caption: [PXL_20241010_142028717],
    ),
    figure(
      image("dataset/PXL_test_broken/PXL_20241010_142051702.jpg"),
      caption: [PXL_20241010_142051702],
    ),
  )
]

#pagebreak()

== Broken Google Pixel 5 images

When running the test against the Pixel 5 images, it was clear pretty quickly that there was an issue.

#pad(y: 20pt)[
  #figure(
    image("docs/prnu_step_9.jpg"),
    caption: [Broken Pixel results],
  ) <broken>
]

Indeed, only one image has a positive correlation with our Pixel 5 PRNU. When looking at the dataset, I noticed that all my training images are horizontal when all my test images are vertical but one. The only horizonal test image is the only one with a positive correlation as shown on @broken.

#pagebreak()

== Fixed Broken Google Pixel 5 images

To fix this, I simply rotated the images to be horizontal and now the result are correct.

#pad(y: 20pt)[
  #figure(
    image("docs/prnu_step_10.jpg"),
    caption: [Broken Pixel results],
  ) <fixed>
]

#pagebreak()

== Random pictures

To be sure we don't have false positives, it's also nice to test all cameras against random pictures from the internet. This time each point for a given color represent an unknown image from the internet, color-coded against the camera it is being tested against.

#pad(y: 20pt)[
  #figure(
    image("docs/prnu_step_11.jpg"),
    caption: [Broken Pixel results],
  )
]

We can see that all the images fall under the same ranges and that there are no outliers, which is normal.

#pagebreak()

= Sources

#let files = ("shared.py", "train.py", "test.py")
#for file in files {
  heading(file, level: 2)

  raw(
    read(file),
    lang: "python",
    block: true,
  )

  if file != "test.py" {
    pagebreak()
  }
}
