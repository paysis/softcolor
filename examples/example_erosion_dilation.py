import numpy as np
import matplotlib.pyplot as plt
from skimage import io

from examples.utils import Timer
from softcolor.morphology import MorphologyInCIELab

if __name__ == "__main__":
    img = io.imread('images/lena-512.gif')
    img = img[100:150, 100:150, :]

    morphology = MorphologyInCIELab()
    se = np.ones(shape=(5, 1), dtype='float32')
    with Timer() as t:
        img_eroded = morphology.erosion(img, structuring_element=se)
        img_dilated = morphology.dilation(img, structuring_element=se)
    print('Time elapsed: {} s'.format(t.interval))

    _, axs = plt.subplots(nrows=2, ncols=2)
    [a.axis('off') for a in axs.flat]
    axs[0, 1].imshow(se)
    axs[0, 0].imshow(img)
    axs[1, 0].imshow(img_eroded)
    axs[1, 1].imshow(img_dilated)
    plt.show()
