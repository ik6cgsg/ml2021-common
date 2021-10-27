import numpy as np
import cv2
import skimage.morphology
import skimage.filters
import skimage.segmentation
import skimage.measure
from typing import List
from intelligent_checker_lib.common import image

TAG = '[common]'


def extract_background_mask(initial_image: np.ndarray, verbose: bool = False):
    img: np.ndarray = initial_image.copy()

    image.show(img, "input img", verbose)

    # extract edges
    img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                  cv2.THRESH_BINARY, 33, 2)
    image.show(img, "adaptive threshold", verbose)

    # remove noise
    img = skimage.morphology.binary_dilation(img, np.ones((3, 3)))
    image.show(img, "binary dilation to remove noise", verbose)

    # close edges
    img = 1 - skimage.morphology.binary_closing(1 - img, np.ones((10, 10)))
    image.show(img, "binary closing of edges", verbose)

    # thicken edges
    img = 1 - skimage.morphology.binary_dilation(1 - img, np.ones((10, 10)))
    image.show(img, "binary erosion to thicken edges", verbose)

    # blur
    img = (skimage.filters.gaussian(img.astype("float"),
                                      sigma=15) * 255).astype("uint8")
    image.show(img, "gaussian", verbose)

    # chan-vese
    img = skimage.segmentation.morphological_chan_vese(img, iterations=30)
    image.show(img, "morphological chan_vese", verbose)

    # extract largest component
    labels = skimage.measure.label(img, 2)
    largestCC = labels == np.argmax(np.bincount(labels.flat))
    img = largestCC.astype("int")
    image.show(img, "largest connected component", verbose)

    # dilate background
    img = skimage.morphology.binary_dilation(img, np.ones((20, 20)))
    image.show(img, "binary dilation to compensate for gauss and chan-vese", verbose)

    return img


def refine_object_mask(img: np.ndarray, initial_object_mask: np.ndarray):
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
        skimage.filters.gaussian(img, 3, preserve_range=False),
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


def cut_out_object(img: np.ndarray, mask: np.ndarray):
    return cv2.bitwise_and(img.astype("uint8") + 1, img.astype("uint8") + 1,
                           mask=mask.astype("uint8"))


def extract_a4_mask(img: np.ndarray) -> np.ndarray:
    background_mask = extract_background_mask(img)
    return refine_object_mask(img, background_mask)


def extract_object_masks(img: np.ndarray) -> List[np.ndarray]:
    background_mask = extract_background_mask(img)
    return [refine_object_mask(img, mask)
            for mask in extract_object_mask_approximations(background_mask)]


def get_perspective_matrix_and_scale(a4mask, verbose: bool = False):
    def order_points(pts):
        rect = np.zeros((4, 2), dtype="float32")
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]
        return rect

    def four_point_transform(pts):
        rect = order_points(pts)
        (tl, tr, br, bl) = rect
        width_a = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
        width_b = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
        max_width = max(int(width_a), int(width_b))
        height_a = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
        height_b = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
        max_height = max(int(height_a), int(height_b))
        dst = np.array([
            [0, 0],
            [max_width - 1, 0],
            [max_width - 1, max_height - 1],
            [0, max_height - 1]], dtype="float32")
        m = cv2.getPerspectiveTransform(rect, dst)
        return m, max_width, max_height

    def get_a4_scale(w, h):
        wk = 297.0 / max(w, h)
        hk = 210.0 / min(w, h)
        return (hk + wk) / 2

    contours, hierarchy = cv2.findContours(a4mask.astype("uint8"), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    perimeter = cv2.arcLength(contours[0], True)
    approx = cv2.approxPolyDP(contours[0], 0.05 * perimeter, True)
    # test
    img = np.zeros(a4mask.shape, "uint8")
    for point in approx:
        x, y = point[0]
        cv2.circle(img, (x, y), 6, (255, 255, 0), 4)
    image.show(img, "4 corner points", verbose)
    pts = np.array([approx[0][0], approx[1][0], approx[2][0], approx[3][0]], dtype="float32")
    # end test
    m, w, h = four_point_transform(pts)
    k = get_a4_scale(w, h)
    if verbose:
        print("Matrix = ", m)
        print("mm in 1 pixel = ", k)
    return m, k, w, h
