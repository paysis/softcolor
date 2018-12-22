from math import ceil

import matplotlib.pyplot as plt
from skimage import io
from skimage.morphology import disk

from examples.utils import Timer
from softcolor.morphology import MorphologyInCIELab, soften_structuring_element

if __name__ == "__main__":
    img = io.imread('images/lena-512.gif')
    img = img[100:200, 100:200, :]

    morphology = MorphologyInCIELab()
    se = disk(3)
    se = soften_structuring_element(se)
    with Timer() as t:
        img_contrasted, img_contrasted_steps = morphology.contrast_mapping_with_steps(img,
                                                                                      structuring_element=se,
                                                                                      num_iterations=3)
    print('Time elapsed contrast mapping: {} s'.format(t.interval))

    _, axs = plt.subplots(nrows=2, ncols=2)
    [a.axis('off') for a in axs.flat]
    axs[0, 1].imshow(se)
    axs[0, 0].imshow(img)
    axs[1, 0].imshow(img_contrasted)
    plt.show()

    _, axs = plt.subplots(nrows=3, ncols=ceil(len(img_contrasted_steps)/3))
    [a.axis('off') for a in axs.flat]
    for idx_step, step in enumerate(img_contrasted_steps):
        axs.flat[idx_step].imshow(step)
    plt.show()
