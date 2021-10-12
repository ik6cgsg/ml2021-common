from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import cv2
import skimage.morphology
import skimage.filters
import skimage.segmentation
import skimage.measure
from skimage.color import label2rgb
import glob
from typing import List


TAG = '[common]'


def do_common_stuff():
    print(f'{TAG} do_common_stuff()')


def extract_background_mask(initial_image: np.ndarray, verbose: bool = False):
    image: np.ndarray = initial_image.copy()

    def show(title: str):
        if verbose:
            plt.figure()
            plt.title(title)
            plt.imshow(image, cmap="gray")

    show("input image")

    # extract edges
    image = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                  cv2.THRESH_BINARY, 33, 2)
    show("adaptive threshold")

    # remove noise
    image = skimage.morphology.binary_dilation(image, np.ones((3, 3)))
    show("binary dilation to remove noise")

    # close edges
    image = 1 - skimage.morphology.binary_closing(1 - image, np.ones((10, 10)))
    show("binary closing of edges")

    # thicken edges
    image = 1 - skimage.morphology.binary_dilation(1 - image, np.ones((10, 10)))
    show("binary erosion to thicken edges")

    # blur
    image = (skimage.filters.gaussian(image.astype("float"),
                                      sigma=15) * 255).astype("uint8")
    show("gaussian")

    # chan-vese
    image = skimage.segmentation.morphological_chan_vese(image, iterations=30)
    show("morphological chan_vese")

    # extract largest component
    labels = skimage.measure.label(image, 2)
    largestCC = labels == np.argmax(np.bincount(labels.flat))
    image = largestCC.astype("int")
    show("largest connected component")

    # dilate background
    image = skimage.morphology.binary_dilation(image, np.ones((20, 20)))
    show("binary dilation to compensate for gauss and chan-vese")

    return image


def refine_object_mask(image: np.ndarray, initial_object_mask: np.ndarray):
    mask = initial_object_mask.copy().astype("int")

    # discard small objects
    mask_size = np.bincount(mask.flat)[1]
    if mask_size < 3000:
        return np.zeros_like(mask)

    # expand mask
    mask = skimage.morphology.binary_dilation(mask, np.ones((15, 15))).astype("int")

    # extract contour
    contour = skimage.measure.find_contours(mask.astype("int"))[0]

    # active contour
    snake: np.ndarray = skimage.segmentation.active_contour(
        skimage.filters.gaussian(image, 3, preserve_range=False),
        contour, alpha=0.015, beta=10, gamma=0.001, max_iterations=500)

    snake[:, [0, 1]] = snake[:, [1, 0]]
    snake = snake.reshape((1, *snake.shape)).astype("int32")

    # convert snake to mask
    mask = np.zeros_like(mask)
    cv2.fillPoly(mask, pts=snake, color=(1, 1, 1))

    return mask


def extract_object_mask_approximations(background_mask) -> List:
    labels = skimage.measure.label(background_mask, True)
    bins = np.bincount(labels.flat)

    output = []
    for i in range(bins.shape[0]):
        output.append((labels == i).astype("int"))

    # TODO: remove object on the edges
    # ...

    return output


def open_image(filename: str):
    img = Image.open(filename).convert("L")
    img.thumbnail((960, 1280))
    return np.array(img).reshape(img.size[1], img.size[0])


def save_image(image: np.ndarray, filename: str):
    Image.fromarray(image.astype("uint8")).save(filename)


def extract_a4_mask(image: np.ndarray) -> np.ndarray:
    background_mask = extract_background_mask(image)
    return refine_object_mask(image, background_mask)


def extract_object_masks(image: np.ndarray) -> List[np.ndarray]:
    background_mask = extract_background_mask(image)
    return [refine_object_mask(image, mask)
            for mask in extract_object_mask_approximations(background_mask)]


if __name__ == "__main__":
    image_names = []
    for infile in glob.glob("BaseJPG\\*.jpg"):
        image_names.append(infile)

    for name in image_names:
        image = open_image(name)

        background = extract_background_mask(image)
        # color = label2rgb(skimage.measure.label(background), image, bg_label=0)
        # save_image(color * 255, "output/" + name + "_background.jpg")
        i = 0
        for mask in extract_object_mask_approximations(background):
            refined_mask = refine_object_mask(image, mask)
            mask_color = label2rgb(skimage.measure.label(mask), image, bg_label=0)
            save_image(mask_color * 255, "output/" + name + str(i) + "_mask.jpg")
            refined_mask_color = label2rgb(skimage.measure.label(refined_mask),
                                           image, bg_label=0)
            save_image(refined_mask_color * 255, "output/" + name + str(i) + "_refined.jpg")
            i += 1
