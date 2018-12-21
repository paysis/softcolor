import matplotlib.pyplot as plt
from skimage import io
from skimage.morphology import disk

from softcolor.morphology import MorphologyInCIELab

if __name__ == "__main__":
    img = io.imread('images/lena-512.gif')

    morphology = MorphologyInCIELab()
    se = disk(3)
    img_eroded = morphology.erosion(img, structuring_element=se)
    img_dilated = morphology.dilation(img, structuring_element=se)

    fig, axs = plt.subplots(nrows=2, ncols=2)
    [a.axis('off') for a in axs.flat]

    axs[0, 1].imshow(se)
    axs[0, 0].imshow(img)
    axs[1, 0].imshow(img_eroded)
    axs[1, 1].imshow(img_dilated)

    plt.show()