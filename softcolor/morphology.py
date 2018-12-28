from math import ceil
from warnings import warn

from scipy.ndimage.filters import convolve
from skimage import color
import numpy as np
from skimage.morphology import disk

from softcolor.aggregation_functions import conjunction_min, r_implication, implication_godel
from softcolor.distance_between_images import euclidean_distance
from softcolor.soft_color_operators import soft_color_erosion, soft_color_dilation


class BaseMorphology:

    def __init__(self, conjunction=None, fuzzy_implication_function=None, distance_images=euclidean_distance):
        if conjunction is None:
            conjunction = conjunction_min
        self.conj = conjunction
        if fuzzy_implication_function is None:
            try:
                fuzzy_implication_function = r_implication(self.conj)
            except AttributeError:
                fuzzy_implication_function = implication_godel
        self.impl = fuzzy_implication_function
        self.dist = distance_images

    def dilation(self, multivariate_image, structuring_element):
        return soft_color_dilation(multivariate_image=multivariate_image,
                                   structuring_element=structuring_element,
                                   fuzzy_conjunction=self.conj)

    def erosion(self, multivariate_image, structuring_element):
        return soft_color_erosion(multivariate_image=multivariate_image,
                                  structuring_element=structuring_element,
                                  fuzzy_implication_function=self.impl)

    def opening(self, multivariate_image, structuring_element):
        return self.dilation(
            self.erosion(
                multivariate_image,
                structuring_element=structuring_element),
            structuring_element=structuring_element[::-1, ::-1]
        )

    def closing(self, multivariate_image, structuring_element):
        return self.erosion(
            self.dilation(
                multivariate_image,
                structuring_element=structuring_element),
            structuring_element=structuring_element[::-1, ::-1]
        )

    def tophat_opening(self, multivariate_image, structuring_element):
        return self.dist(
            multivariate_image,
            self.opening(multivariate_image, structuring_element=structuring_element)
        )

    def tophat_closing(self, multivariate_image, structuring_element):
        return self.dist(
            multivariate_image,
            self.closing(multivariate_image, structuring_element=structuring_element)
        )

    def gradient(self, multivariate_image, structuring_element):
        """ Distance between erosion and dilation of the image. """
        return self.dist(
            self.erosion(multivariate_image, structuring_element=structuring_element),
            self.dilation(multivariate_image, structuring_element=structuring_element)
        )

    def inpaint_with_steps(self, multivariate_image, structuring_element, max_iterations=10):
        """ Iteratively recover pixels given by 0.5 * (opening + closing). """
        inpainted_image = multivariate_image
        steps = [inpainted_image.copy()]
        mask_unknown = np.isnan(inpainted_image[:, :, 0])
        idx_it = 0
        while np.any(mask_unknown) and idx_it <= max_iterations:
            closing = self.closing(inpainted_image,
                                   structuring_element=structuring_element)
            opening = self.opening(inpainted_image,
                                   structuring_element=structuring_element)
            mask_recovered = mask_unknown & ~np.isnan(closing[:, :, 0]) & ~np.isnan(opening[:, :, 0])
            if not np.any(mask_recovered):
                break
            mask_recovered = np.tile(mask_recovered[:, :, np.newaxis], (1, 1, 3))
            inpainted_image[mask_recovered] = 0.5 * (closing[mask_recovered] + opening[mask_recovered])
            mask_unknown = np.isnan(inpainted_image[:, :, 0])
            idx_it += 1
            steps += [inpainted_image.copy()]
        return inpainted_image, steps

    def inpaint(self, multivariate_image, structuring_element, max_iterations=10):
        """ Iteratively recover pixels given by 0.5 * (opening + closing). """
        inpainted_image, _ = self.inpaint_with_steps(multivariate_image,
                                                     structuring_element=structuring_element,
                                                     max_iterations=max_iterations)
        return inpainted_image

    def contrast_mapping_with_steps(self, multivariate_image, structuring_element, num_iterations=3):
        """ Iteratively change pixels as the most similar one between their dilation and their erosion. """
        contrasted_image = multivariate_image
        idx_it = 0
        steps = [contrasted_image.copy()]
        while idx_it <= num_iterations:
            dilation = self.dilation(multivariate_image,
                                     structuring_element=structuring_element)
            erosion = self.erosion(multivariate_image,
                                   structuring_element=structuring_element)
            d_dilation = self.dist(multivariate_image, dilation)
            d_erosion = self.dist(multivariate_image, erosion)
            mask_dilation_is_closest = d_dilation < d_erosion
            mask_dilation_is_closest = np.tile(mask_dilation_is_closest[:, :, np.newaxis], (1, 1, 3))
            contrasted_image[mask_dilation_is_closest] = dilation[mask_dilation_is_closest]
            contrasted_image[~mask_dilation_is_closest] = erosion[~mask_dilation_is_closest]
            idx_it += 1
            steps += [contrasted_image.copy()]
        return contrasted_image, steps

    def contrast_mapping(self, multivariate_image, structuring_element, num_iterations=10):
        """ Iteratively change pixels as the most similar one between their dilation and their erosion. """
        contrasted_image, _ = self.contrast_mapping_with_steps(
            multivariate_image,
            structuring_element=structuring_element,
            num_iterations=num_iterations,
        )
        return contrasted_image


class MorphologyInCIELab(BaseMorphology):

    def dilation(self, image_as_rgb, structuring_element):
        lab_image = _rgb_to_lab(image_as_rgb)/100.0
        lab_dilation = super().dilation(multivariate_image=lab_image,
                                        structuring_element=structuring_element) * 100.0
        return _lab_to_rgb(lab_dilation)

    def erosion(self, image_as_rgb, structuring_element):
        lab_image = _rgb_to_lab(image_as_rgb)/100.0
        lab_erosion = super().erosion(multivariate_image=lab_image,
                                      structuring_element=structuring_element) * 100.0
        return _lab_to_rgb(lab_erosion)


def _rgb_to_lab(image_as_rgb):
    # Wrapper of skimage.color.rgb2lab to avoid computing on NaN values
    rgb_flat = image_as_rgb.reshape((-1, 1, image_as_rgb.shape[2]))
    nonnan_mask = ~np.isnan(rgb_flat[:, 0, 0])
    lab_flat = np.full(shape=rgb_flat.shape,
                       dtype='float64',
                       fill_value=np.nan)
    lab_flat[nonnan_mask, :, :] = color.rgb2lab(rgb_flat[nonnan_mask, :, :])
    lab = lab_flat.reshape(image_as_rgb.shape)
    return lab


def _lab_to_rgb(image_as_lab):
    # Wrapper of skimage.color.lab2rgb to avoid computing on NaN values
    lab_flat = image_as_lab.reshape((-1, 1, image_as_lab.shape[2]))
    nonnan_mask = ~np.isnan(lab_flat[:, 0, 0])
    rgb_flat = np.full(shape=lab_flat.shape,
                       dtype='float64',
                       fill_value=np.nan)
    rgb_flat[nonnan_mask, :, :] = color.lab2rgb(lab_flat[nonnan_mask, :, :])
    rgb = rgb_flat.reshape(image_as_lab.shape)
    return rgb


def soften_structuring_element(structuring_element, sz_averaging_kernel_in_px=None, preserve_shape=True):
    if sz_averaging_kernel_in_px is not None:
        pad_px = sz_averaging_kernel_in_px//2
    else:
        pad_px = ceil(sum(structuring_element.shape)/10)
        pad_px = max(1, pad_px)

    if np.all(structuring_element <= 1):
        structuring_element = structuring_element.astype('float32')

    se_padded = np.pad(structuring_element,
                       pad_width=((pad_px, pad_px), (pad_px, pad_px)),
                       mode='constant', constant_values=0)

    kernel = disk(pad_px)
    kernel = convolve(input=kernel, weights=kernel, mode='constant', cval=0.0)
    kernel = kernel/np.sum(kernel)
    se_soft = convolve(input=se_padded, weights=kernel, mode='constant', cval=0.0)

    se_orig_center = tuple(e//2 for e in structuring_element.shape)
    se_soft_center = tuple(e//2 for e in se_soft.shape)
    if structuring_element[se_orig_center] == 1 and se_soft[se_soft_center] != 1:
        msg = ("""
        The original Structuring Element had a center equal to 1, but the softened one does not.
        This may produce unexpected behaviours.""")
        warn(RuntimeWarning(msg))

    if preserve_shape:
        se_soft = se_soft[pad_px:-pad_px, pad_px:-pad_px]

    return se_soft



