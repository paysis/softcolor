from numba import jit
import numpy as np


def soft_color_erosion(multivariate_image, structuring_element, fuzzy_implication_function):
    return _base_soft_color_operator(
        multivariate_image=multivariate_image,
        structuring_element=structuring_element,
        se_distances_wrt_center=_get_distances_wrt_center(structuring_element.shape),
        aggregation_function=fuzzy_implication_function,
        order_criteria_1=np.min,
        order_criteria_3=np.min,
    )


def soft_color_dilation(multivariate_image, structuring_element, fuzzy_conjunction):
    return _base_soft_color_operator(
        multivariate_image=multivariate_image,
        structuring_element=structuring_element,
        se_distances_wrt_center=_get_distances_wrt_center(structuring_element.shape),
        aggregation_function=fuzzy_conjunction,
        order_criteria_1=np.max,
        order_criteria_3=np.max,
    )


def _base_soft_color_operator(multivariate_image, structuring_element, se_distances_wrt_center, aggregation_function,
                              order_criteria_1, order_criteria_3):
    num_channels = multivariate_image.shape[2]
    sz_im = multivariate_image.shape[:2]
    sz_se = structuring_element.shape
    se_center_idcs = [e//2 for e in sz_se]
    se_before_center_excluded = se_center_idcs
    se_after_center_included = [e1 - e2 for e1, e2 in zip(sz_se, se_center_idcs)]
    se_after_center_excluded = [e-1 for e in se_after_center_included]

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

            im_values = multivariate_image[im_ini_i:im_end_i, im_ini_j:im_end_j, :].copy()
            se_values = structuring_element[se_ini_i:se_end_i, se_ini_j:se_end_j].copy()

            computed_values = np.concatenate((aggregation_function(se_values, im_values[:, :, 0])[:, :, np.newaxis], im_values[:, :, 1:]), axis=2)
            computed_values_flattened = computed_values.reshape((-1, num_channels))

            optm_idcs_nonnan = ~np.isnan(computed_values_flattened[:, 0])
            if all(optm_idcs_nonnan == False):
                eroded_image[i, j, :] = np.nan

            else:
                computed_values_flattened = computed_values_flattened[optm_idcs_nonnan, :]

                optm_value = order_criteria_1(computed_values_flattened[:, 0], axis=0)
                optm_idcs_criteria_1 = (computed_values_flattened[:, 0] == optm_value)
                if optm_idcs_criteria_1.size > 1:
                    d_center_flattened = se_distances_wrt_center[se_ini_i:se_end_i, se_ini_j:se_end_j].flatten()

                    d_center_flattened = d_center_flattened[optm_idcs_criteria_1]

                    optimal_distance = np.min(d_center_flattened, axis=0)
                    optm_idcs_criteria_2 = (d_center_flattened == optimal_distance)
                    if optm_idcs_criteria_2.size != 1:
                        computed_values_flattened = computed_values_flattened[optm_idcs_criteria_1, :]
                        computed_values_flattened = computed_values_flattened[optm_idcs_criteria_2, :]

                        best_idx = 0
                        best_value = computed_values_flattened[best_idx, :]
                        for current_idx in range(1, computed_values_flattened.shape[0]):
                            current_value = computed_values_flattened[current_idx, :]
                            idcs_nonmatching_element = np.where(best_value != current_value)[0]
                            if len(idcs_nonmatching_element) != 0 and order_criteria_3(current_value[idcs_nonmatching_element[0]], best_value[idcs_nonmatching_element[0]]):
                                best_idx = current_idx
                                best_value = current_value

                        # Select only best_idx
                        optm_idcs_criteria_2[:] = False
                        optm_idcs_criteria_2[best_idx] = True

                    # Modify only idcs being True (the ones on tie resolution)
                    optm_idcs_criteria_1[optm_idcs_criteria_1] = optm_idcs_criteria_2
                # Modify only idcs being True (the ones on tie resolution)
                optm_idcs_nonnan[optm_idcs_nonnan] = optm_idcs_criteria_1

                sel_idcs = np.where(optm_idcs_nonnan)[0]
                sel_i, sel_j = np.unravel_index(sel_idcs[0], im_values.shape[:2])
                eroded_image[i, j, :] = computed_values[sel_i, sel_j, :]
    return eroded_image


def _get_distances_wrt_center(spatial_shape):
    center_idcs = [e// 2 for e in spatial_shape]
    i = np.tile(np.arange(-center_idcs[0], spatial_shape[0]-center_idcs[0])[:, np.newaxis], (1, spatial_shape[1]))
    j = np.tile(np.arange(-center_idcs[1], spatial_shape[1]-center_idcs[1])[np.newaxis, :], (spatial_shape[0], 1))
    coordinates = np.concatenate((i[:, :, np.newaxis], j[:, :, np.newaxis]), axis=2)
    return np.linalg.norm(coordinates, axis=2)
