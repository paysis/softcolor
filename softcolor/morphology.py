import cv2
import numpy as np

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
            multivariate_image=self.erosion(
                multivariate_image=multivariate_image,
                structuring_element=structuring_element),
            structuring_element=structuring_element[::-1, ::-1]
        )

    def closing(self, multivariate_image, structuring_element):
        return self.erosion(
            multivariate_image=self.dilation(
                multivariate_image=multivariate_image,
                structuring_element=structuring_element),
            structuring_element=structuring_element[::-1, ::-1]
        )

    def tophat_opening(self, multivariate_image, structuring_element):
        return self.dist(
            multivariate_image,
            self.opening(multivariate_image=multivariate_image, structuring_element=structuring_element)
        )

    def tophat_closing(self, multivariate_image, structuring_element):
        return self.dist(
            multivariate_image,
            self.closing(multivariate_image=multivariate_image, structuring_element=structuring_element)
        )

    def gradient(self, multivariate_image, structuring_element):
        """ Distance between erosion and dilation of the image. """
        return self.dist(
            self.erosion(multivariate_image=multivariate_image, structuring_element=structuring_element),
            self.dilation(multivariate_image=multivariate_image, structuring_element=structuring_element)
        )

    def inpaint(self, multivariate_image, structuring_element, max_iterations=100):
        """ Iteratively recover pixels given by 0.5 * (opening + closing). """
        inpainted_image = multivariate_image
        mask_unknown = np.isnan(inpainted_image[:, :, 0])
        idx_it = 0
        while np.any(mask_unknown) and idx_it <= max_iterations:
            closing = self.closing(multivariate_image=inpainted_image,
                                   structuring_element=structuring_element)
            opening = self.opening(multivariate_image=inpainted_image,
                                   structuring_element=structuring_element)
            mask_recovered = mask_unknown & ~closing[:, :, 0].isnan() & ~opening[:, :, 0].isnan()
            if not np.any(mask_recovered):
                break
            mask_recovered = np.tile(mask_recovered[:, :, np.newaxis], (1, 1, 3))
            inpainted_image[mask_recovered] = 0.5 * (closing[mask_recovered] + opening[mask_recovered])
            mask_unknown = np.isnan(inpainted_image[:, :, 0])
            idx_it += 1
        return inpainted_image


    def contrast_mapping(self, multivariate_image, structuring_element, num_iterations=10):
        """ Iteratively change pixels as the most similar one between their dilation and their erosion. """
        contrasted_image = multivariate_image
        idx_it = 0
        while idx_it <= num_iterations:
            dilation = self.dilation(multivariate_image=multivariate_image,
                                     structuring_element=structuring_element)
            erosion = self.erosion(multivariate_image=multivariate_image,
                                   structuring_element=structuring_element)
            d_dilation = self.dist(multivariate_image, dilation)
            d_erosion = self.dist(multivariate_image, erosion)
            mask_dilation_is_closest = d_dilation < d_erosion
            mask_dilation_is_closest = np.tile(mask_dilation_is_closest[:, :, np.newaxis], (1, 1, 3))
            contrasted_image[mask_dilation_is_closest] = dilation[mask_dilation_is_closest]
            contrasted_image[~mask_dilation_is_closest] = erosion[~mask_dilation_is_closest]
        return contrasted_image


class MorphologyInCIELab(BaseMorphology):

    def dilation(self, image_as_bgr, structuring_element):
        lab_image = cv2.cvtColor(image_as_bgr, cv2.COLOR_BGR2LAB)
        lab_dilation = soft_color_dilation(multivariate_image=lab_image,
                                           structuring_element=structuring_element,
                                           fuzzy_conjunction=self.conj)
        return cv2.cvtColor(lab_dilation, cv2.COLOR_LAB2BGR)

    def erosion(self, image_as_bgr, structuring_element):
        lab_image = cv2.cvtColor(image_as_bgr, cv2.COLOR_BGR2LAB)
        lab_erosion = soft_color_erosion(multivariate_image=lab_image,
                                         structuring_element=structuring_element,
                                         fuzzy_implication_function=self.impl)
        return cv2.cvtColor(lab_erosion, cv2.COLOR_LAB2BGR)

