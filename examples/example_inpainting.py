from math import ceil

import numpy as np
import matplotlib.pyplot as plt
from skimage import io, img_as_float
from skimage.morphology import disk

from examples.utils import Timer
from softcolor.morphology import MorphologyInCIELab, soften_structuring_element


def _replace_nans_with_white(image_as_rgb):
    nan_mask = np.isnan(image_as_rgb[:, :, 0])
    img_result = np.ones_like(image_as_rgb)
    for idx_c in range(image_as_rgb.shape[2]):
        channel = image_as_rgb[:, :, idx_c]
        channel[nan_mask] = 1
        img_result[:, :, idx_c] = channel
    return img_result


if __name__ == "__main__":
    img = io.imread('images/lena-512.gif')
    img = img[100:150, 100:150, :]
    img = img_as_float(img)

    probability_forgetting_pixel = 0.95
    nan_mask = np.random.choice([True, False], img.shape[:2],
                                p=[probability_forgetting_pixel, 1 - probability_forgetting_pixel])
    for idx_c in range(img.shape[2]):
        channel = img[:, :, idx_c]
        channel[nan_mask] = np.nan
        img[:, :, idx_c] = channel

    morphology = MorphologyInCIELab()
    se = disk(1)
    with Timer() as t:
        img_inpainted, img_inpainted_steps = morphology.inpaint_with_steps(img,
                                                                           structuring_element=se,
                                                                           max_iterations=10)
    print('Time elapsed inpainting: {} s'.format(t.interval))

    _, axs = plt.subplots(nrows=2, ncols=2)
    [a.axis('off') for a in axs.flat]
    axs[0, 0].imshow(_replace_nans_with_white(img))
    axs[0, 1].imshow(se)
    axs[1, 0].imshow(img_inpainted)
    plt.show()

    _, axs = plt.subplots(nrows=3, ncols=ceil(len(img_inpainted_steps)/3))
    [a.axis('off') for a in axs.flat]
    for idx_step, step in enumerate(img_inpainted_steps):
        axs.flat[idx_step].imshow(_replace_nans_with_white(step))
    plt.show()
