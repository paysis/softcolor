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
    img_original = io.imread('images/lena-512.gif')
    img_with_nans = img_as_float(img_original.copy())

    probability_forgetting_pixel = 0.75
    nan_mask = np.random.choice([True, False], img_with_nans.shape[:2],
                                p=[probability_forgetting_pixel, 1 - probability_forgetting_pixel])
    for idx_c in range(img_with_nans.shape[2]):
        channel = img_with_nans[:, :, idx_c]
        channel[nan_mask] = np.nan
        img_with_nans[:, :, idx_c] = channel

    morphology = MorphologyInCIELab()
    se = disk(1).astype('float64')
    se[se == 0] = np.nan
    with Timer() as t:
        img_inpainted, img_inpainted_steps = morphology.inpaint_with_steps(img_with_nans,
                                                                           structuring_element=se,
                                                                           max_iterations=10)
    print('Time elapsed inpainting: {} s'.format(t.interval))

    _, axs = plt.subplots(nrows=2, ncols=2)
    [a.axis('off') for a in axs.flat]
    axs[0, 0].imshow(img_original)
    axs[0, 1].imshow(se)
    axs[1, 0].imshow(_replace_nans_with_white(img_with_nans))
    axs[1, 1].imshow(img_inpainted)
    plt.show()

    _, axs = plt.subplots(nrows=2, ncols=ceil(len(img_inpainted_steps)/2))
    [a.axis('off') for a in axs.flat]
    for idx_step, step in enumerate(img_inpainted_steps):
        axs.flat[idx_step].imshow(_replace_nans_with_white(step))
    plt.show()
