import matplotlib.pyplot as plt
from skimage import io
from skimage.morphology import disk

from examples.utils import Timer
from softcolor.morphology import MorphologyInCIELab, soften_structuring_element

if __name__ == "__main__":
    img = io.imread('images/dermoscopy.jpg')

    morphology = MorphologyInCIELab()
    se = disk(5)
    se = soften_structuring_element(se)
    with Timer() as t:
        img_tophat_opening = morphology.tophat_opening(img, structuring_element=se)
        img_tophat_closing = morphology.tophat_closing(img, structuring_element=se)
    print('Time elapsed: {} s'.format(t.interval))

    fig, axs = plt.subplots(nrows=2, ncols=2)
    [a.axis('off') for a in axs.flat]
    axs[0, 1].imshow(se)
    axs[0, 0].imshow(img)
    axs[1, 0].imshow(img_tophat_opening)
    axs[1, 1].imshow(img_tophat_closing)
    fig.show()
