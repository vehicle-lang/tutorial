"""Generate IDX data set files for the Chapter 3 robustness exercises.

Vehicle expects the two data sets declared in `mnist-robustness.vcl` in the
same format as the `t2-images.idx` / `t2-labels.idx` pair used in the chapter:

  images  float64 IDX tensor of shape [n, 28, 28], pixel values in [0, 1]
  labels  uint8   IDX vector of shape [n]

The pixel range matters: the specification's `validImage` predicate requires
`0 <= x ! i ! j <= 1`, so the raw 0-255 values are divided by 255.

MNIST is itself distributed in the IDX format, so this script downloads the
official test set and re-emits prefixes of it at the sizes you ask for. The
only dependencies are `numpy` and `idx2numpy`:

    pip install numpy idx2numpy

Usage:

    python create_dataset.py             # writes t2, t10 and t100 pairs
    python create_dataset.py 2 50 500    # writes the sizes you ask for

Larger data sets make Exercise #3 slower but its statistics more meaningful.
"""

import gzip
import sys
import urllib.request
from pathlib import Path

import idx2numpy
import numpy as np

DEFAULT_SIZES = [2, 10, 100]
CACHE_DIR = Path("mnist-data")
MIRROR = "https://ossci-datasets.s3.amazonaws.com/mnist"
IMAGE_ARCHIVE = "t10k-images-idx3-ubyte.gz"
LABEL_ARCHIVE = "t10k-labels-idx1-ubyte.gz"


def download(archive):
    """Fetch `archive` from the MNIST mirror, caching it under CACHE_DIR."""
    CACHE_DIR.mkdir(exist_ok=True)
    target = CACHE_DIR / archive
    if not target.exists():
        print(f"downloading {archive}")
        urllib.request.urlretrieve(f"{MIRROR}/{archive}", target)
    return target


def load_mnist_test_set():
    """Return the MNIST test set as (images in [0,1] float64, labels uint8)."""
    with gzip.open(download(IMAGE_ARCHIVE)) as f:
        images = idx2numpy.convert_from_string(f.read()).astype(np.float64) / 255.0
    with gzip.open(download(LABEL_ARCHIVE)) as f:
        labels = idx2numpy.convert_from_string(f.read()).astype(np.uint8)
    return images, labels


def write_dataset(images, labels, n):
    """Write the first `n` images and labels as an IDX pair."""
    if n > len(images):
        raise ValueError(f"asked for {n} images but the test set has only {len(images)}")
    image_file, label_file = f"t{n}-images.idx", f"t{n}-labels.idx"
    idx2numpy.convert_to_file(image_file, images[:n])
    idx2numpy.convert_to_file(label_file, labels[:n])
    print(f"wrote {image_file} {images[:n].shape} {images.dtype}"
          f" and {label_file} {labels[:n].shape} {labels.dtype}")


if __name__ == "__main__":
    sizes = [int(arg) for arg in sys.argv[1:]] or DEFAULT_SIZES
    images, labels = load_mnist_test_set()
    for n in sizes:
        write_dataset(images, labels, n)
