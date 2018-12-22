from math import ceil

import numpy as np
import matplotlib.pyplot as plt
from skimage import io, img_as_float
from skimage.morphology import disk

from examples.utils import Timer
from softcolor.morphology import MorphologyInCIELab, soften_structuring_element

if __name__ == "__main__":
    img = io.imread('images/lena-512.gif')
    img = img[100:200, 100:200, :]
    img = img_as_float(img)

    probability_forgetting_pixel = 0.5
    nan_mask = np.random.choice([True, False], img.shape[:2],
                                p=[probability_forgetting_pixel, 1 - probability_forgetting_pixel])
    for idx_c in range(img.shape[2]):
        channel = img[:, :, idx_c]
        channel[nan_mask] = np.nan
        img[:, :, idx_c] = channel

    morphology = MorphologyInCIELab()
    se = disk(3)
    se = soften_structuring_element(se)
    with Timer() as t:
        img_inpainted, img_inpainted_steps = morphology.inpaint_with_steps(img,
                                                                           structuring_element=se,
                                                                           max_iterations=10)
    print('Time elapsed inpainting: {} s'.format(t.interval))

    _, axs = plt.subplots(nrows=2, ncols=2)
    [a.axis('off') for a in axs.flat]
    axs[0, 1].imshow(se)
    axs[0, 0].imshow(img)
    axs[1, 0].imshow(img_inpainted)
    plt.show()

    _, axs = plt.subplots(nrows=3, ncols=ceil(len(img_inpainted_steps)/3))
    [a.axis('off') for a in axs.flat]
    for idx_step, step in enumerate(img_inpainted_steps):
        axs.flat[idx_step].imshow(step)
    plt.show()

