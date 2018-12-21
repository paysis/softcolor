import matplotlib.pyplot as plt
from skimage import io
from skimage.morphology import disk

from softcolor.morphology import MorphologyInCIELab

if __name__=="__main__":
    img = io.imread('images/lena-512.jpg')

    morphology = MorphologyInCIELab()
    img_eroded = morphology.erosion(img, structuring_element=disk(5))
    img_dilated = morphology.dilation(img, structuring_element=disk(5))

    fig, axis = plt.subplots(nrows=2, ncols=3)

    axis[0].axis("off")
    axis[0].imshow(img_eroded)

    axis[1].axis("off")
    axis[1].imshow(img)

    axis[2].axis("off")
    axis[2].imshow(img)

    plt.show()