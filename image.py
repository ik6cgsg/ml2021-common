from PIL import Image
import numpy as np
import matplotlib.pyplot as plt


def open_image(filename: str):
    img = Image.open(filename).convert("L")
    img.thumbnail((960, 1280))
    return np.array(img).reshape(img.size[1], img.size[0])


def save_image(image: np.ndarray, filename: str):
    Image.fromarray(image.astype("uint8")).save(filename)


def open_image_rgb(filename: str):
    img: Image = Image.open(filename)
    img.thumbnail((960, 1280))
    return np.array(img).reshape(img.size[1], img.size[0], 3)


def show(img: np.ndarray, title: str, verbose: bool = True, cmap: str = "gray"):
    if verbose:
        plt.figure()
        plt.title(title)
        plt.imshow(img, cmap='gray')
        plt.show()
