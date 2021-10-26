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
plt.interactive(False)


def do_common_stuff():
    print(f'{TAG} do_common_stuff()')


def show(img: np.ndarray, title: str, verbose: bool = False):
    if verbose:
        plt.figure()
        plt.title(title)
        plt.imshow(img, cmap="gray")
        plt.show()


def extract_background_mask(initial_image: np.ndarray, verbose: bool = False):
    image: np.ndarray = initial_image.copy()

    show(image, "input image", verbose)

    # extract edges
    image = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                  cv2.THRESH_BINARY, 33, 2)
    show(image, "adaptive threshold", verbose)

    # remove noise
    image = skimage.morphology.binary_dilation(image, np.ones((3, 3)))
    show(image, "binary dilation to remove noise", verbose)

    # close edges
    image = 1 - skimage.morphology.binary_closing(1 - image, np.ones((10, 10)))
    show(image, "binary closing of edges", verbose)

    # thicken edges
    image = 1 - skimage.morphology.binary_dilation(1 - image, np.ones((10, 10)))
    show(image, "binary erosion to thicken edges", verbose)

    # blur
    image = (skimage.filters.gaussian(image.astype("float"),
                                      sigma=15) * 255).astype("uint8")
    show(image, "gaussian", verbose)

    # chan-vese
    image = skimage.segmentation.morphological_chan_vese(image, iterations=30)
    show(image, "morphological chan_vese", verbose)

    # extract largest component
    labels = skimage.measure.label(image, 2)
    largestCC = labels == np.argmax(np.bincount(labels.flat))
    image = largestCC.astype("int")
    show(image, "largest connected component", verbose)

    # dilate background
    image = skimage.morphology.binary_dilation(image, np.ones((20, 20)))
    show(image, "binary dilation to compensate for gauss and chan-vese", verbose)

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

    def four_point_transform(image, pts):
        rect = order_points(pts)
        (tl, tr, br, bl) = rect
        widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
        widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
        maxWidth = max(int(widthA), int(widthB))
        heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
        heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
        maxHeight = max(int(heightA), int(heightB))
        dst = np.array([
            [0, 0],
            [maxWidth - 1, 0],
            [maxWidth - 1, maxHeight - 1],
            [0, maxHeight - 1]], dtype="float32")
        M = cv2.getPerspectiveTransform(rect, dst)
        wk = 297.0 / max(maxWidth, maxHeight)
        hk = 210.0 / min(maxWidth, maxHeight)
        k = (hk + wk) / 2
        warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
        show(warped, "transformed", verbose)
        return M, k

    contours, hierarchy = cv2.findContours(a4mask.astype("uint8"), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    perimeter = cv2.arcLength(contours[0], True)
    approx = cv2.approxPolyDP(contours[0], 0.05 * perimeter, True)
    # test
    img = np.zeros(a4mask.shape, "uint8")
    for point in approx:
        x, y = point[0]
        cv2.circle(img, (x, y), 6, (255, 255, 0), 4)
    pts = np.array([approx[0][0], approx[1][0], approx[2][0], approx[3][0]], dtype="float32")
    ordered = order_points(pts)
    (tl, tr, br, bl) = ordered
    cv2.line(img, (tl[0].astype("int"), tl[1].astype("int")), (br[0].astype("int"), br[1].astype("int")),
             (255, 255, 0), thickness=3)
    cv2.line(img, (tr[0].astype("int"), tr[1].astype("int")), (bl[0].astype("int"), bl[1].astype("int")),
             (255, 255, 0), thickness=3)
    show(img, "cross", verbose)
    # end test
    M, k = four_point_transform(img, pts)
    print("Matrix = ", M)
    print("mm in 1 pixel = ", k)
    return M, k


def open_image_rgb(filename: str):
    img: Image = Image.open(filename)
    img.thumbnail((960, 1280))
    return np.array(img).reshape(img.size[1], img.size[0], 3)


if __name__ == "__main__":
    image = open_image("a4.jpg")
    background = extract_background_mask(image, False)
    show(background, "mask", True)
    m, k = get_perspective_matrix_and_scale(background, True)
    # image_names = []
    # for infile in glob.glob("BaseJPG\\*.jpg"):
    #     image_names.append(infile)
    #
    # for name in image_names:
    #     image = open_image(name)
    #
    #     background = extract_background_mask(image)
    #     # color = label2rgb(skimage.measure.label(background), image, bg_label=0)
    #     # save_image(color * 255, "output/" + name + "_background.jpg")
    #     i = 0
    #     for mask in extract_object_mask_approximations(background):
    #         refined_mask = refine_object_mask(image, mask)
    #         mask_color = label2rgb(skimage.measure.label(mask), image, bg_label=0)
    #         save_image(mask_color * 255, "output/" + name + str(i) + "_mask.jpg")
    #         refined_mask_color = label2rgb(skimage.measure.label(refined_mask),
    #                                        image, bg_label=0)
    #         save_image(refined_mask_color * 255, "output/" + name + str(i) + "_refined.jpg")
    #         i += 1
