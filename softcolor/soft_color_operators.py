from numba import jit
import numpy as np


# @jit(nopython=True)
def soft_color_erosion(multivariate_image, structuring_element, fuzzy_implication_function):
    num_channels = multivariate_image.shape[2]
    sz_im = multivariate_image.shape[:2]
    sz_se = structuring_element.shape
    se_center_idcs = [e//2 for e in sz_se]
    se_before_center_excluded = se_center_idcs
    se_after_center_included = [e1 - e2 for e1, e2 in zip(sz_se, se_center_idcs)]
    se_after_center_excluded = [e-1 for e in se_after_center_included]
    se_distances_wrt_center = _get_distances_wrt_center(sz_se)

    eroded_image = np.zeros(shape=multivariate_image.shape)
    for i in range(multivariate_image.shape[0]):
        im_ini_i = max(0, i-se_before_center_excluded[0])
        im_end_i = min(sz_im[0], i+se_after_center_included[0])
        se_ini_i = max(0, se_before_center_excluded[0]-i)
        se_end_i = min(sz_se[0], se_after_center_excluded[0]-i+sz_im[0])

        for j in range(multivariate_image.shape[1]):
            im_ini_j = max(0, j-se_before_center_excluded[1])
            im_end_j = min(sz_im[1], j+se_after_center_included[1])
            se_ini_j = max(0, se_before_center_excluded[1]-j)
            se_end_j = min(sz_se[1], se_after_center_excluded[1]-j+sz_im[1])

            im_values = multivariate_image[im_ini_i:im_end_i, im_ini_j:im_end_j, :]
            se_values = structuring_element[se_ini_i:se_end_i, se_ini_j:se_end_j]

            computed_values = np.concatenate((fuzzy_implication_function(se_values, im_values[:, :, 0])[:, :, np.newaxis], im_values[:, :, 1:]), axis=2)
            computed_values_flattened = computed_values.reshape((-1, num_channels))

            optm_value = np.nanmin(computed_values_flattened[:, 0], axis=0)
            optm_idcs_criteria_1 = (computed_values_flattened[:, 0] == optm_value)
            if optm_idcs_criteria_1.size != 1:
                d_center_flattened = se_distances_wrt_center[se_ini_i:se_end_i, se_ini_j:se_end_j].flatten()

                computed_values_flattened = computed_values_flattened[optm_idcs_criteria_1, :]
                d_center_flattened = d_center_flattened[optm_idcs_criteria_1]

                optimal_distance = np.min(d_center_flattened, axis=0)
                optm_idcs_criteria_2 = (d_center_flattened == optimal_distance)
                if optm_idcs_criteria_2.size != 1:
                    # TODO: finish by implementing tie resolving based on lexicographical order
                    pass

                # Modify ony idcs being True (that were thus passed on to tie resolution)
                optm_idcs_criteria_1[optm_idcs_criteria_1] = optm_idcs_criteria_2

            sel_idcs = np.where(optm_idcs_criteria_1)[0]
            sel_i, sel_j = np.unravel_index(sel_idcs[0], im_values.shape[:2])
            assert multivariate_image[i, j, 0] < computed_values[sel_i, sel_j, 0]
            eroded_image[i, j, :] = computed_values[sel_i, sel_j, :]


    return eroded_image


def soft_color_dilation(multivariate_image, structuring_element, fuzzy_conjunction):
    # TODO: implement
    return multivariate_image


def _get_distances_wrt_center(spatial_shape):
    center_idcs = [e// 2 for e in spatial_shape]
    i = np.tile(np.arange(-center_idcs[0], spatial_shape[0]-center_idcs[0])[:, np.newaxis], (1, spatial_shape[1]))
    j = np.tile(np.arange(-center_idcs[1], spatial_shape[1]-center_idcs[1])[np.newaxis, :], (spatial_shape[0], 1))
    coordinates = np.concatenate((i[:, :, np.newaxis], j[:, :, np.newaxis]), axis=2)
    return np.linalg.norm(coordinates, axis=2)
