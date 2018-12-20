import numpy as np


def soft_color_erosion(multivariate_image, structuring_element, fuzzy_implication_function):
    num_channels = multivariate_image.shape[2]
    sz_im = multivariate_image.shape[:2]
    sz_se = structuring_element.shape
    se_center_idcs = sz_se//2
    se_before_center_excluded = se_center_idcs
    se_after_center_included = sz_se - se_center_idcs
    se_distances_wrt_center = _get_distances_wrt_center(sz_se)

    eroded_image = np.zeros(shape=multivariate_image.shape)
    for i in range(multivariate_image.shape[0]):
        im_ini_i = max(0, i-se_before_center_excluded[0])
        im_end_i = min(sz_im[0], i+se_after_center_included[0])
        se_ini_i = max(0, se_before_center_excluded[0]-i)
        se_end_i = min(sz_se[0], i+se_after_center_included[0]-sz_im[0])

        for j in range(multivariate_image.shape[1]):
            im_ini_j = max(0, j-se_before_center_excluded[1])
            im_end_j = min(sz_im[1], j+se_after_center_included[1])
            se_ini_j = max(0, se_before_center_excluded[1]-j)
            se_end_j = min(sz_se[1], j+se_after_center_included[1]-sz_im[1])

            im_values = multivariate_image[im_ini_i:im_end_i, im_ini_j:im_end_j, :]
            se_values = structuring_element[se_ini_i:se_end_i, se_ini_j:se_end_j]

            computed_values = np.concatenate((fuzzy_implication_function(se_values, im_values), im_values[:, :, 1:]))
            computed_values_flattened = computed_values.reshape(shape=(-1, num_channels))

            optimal_value = np.nanmin(computed_values_flattened[:, 0], axis=0)
            selected_idcs = (computed_values[:, 0] == optimal_value)
            if selected_idcs.size != 1:
                d_center = se_distances_wrt_center[se_ini_i:se_end_i, se_ini_j:se_end_j].reshape(shape=(-1, num_channels))

                computed_values_flattened = computed_values_flattened[selected_idcs]
                d_center_flattened = d_center[selected_idcs]

                optimal_distance = np.min(d_center_flattened, axis=0)
                selected_idcs = (d_center_flattened[:, 0] == optimal_distance)
                if selected_idcs.size != 1:
                    selected_idcs = np.min()

            sel_i, sel_j = np.unravel_index(selected_idcs[0], im_values.shape)
            eroded_image[i, j, :] = computed_values[i, j, :]


    return eroded_image


def soft_color_dilation(multivariate_image, structuring_element, fuzzy_conjunction):
    pass


def _get_distances_wrt_center(spatial_shape):
    center_idcs = spatial_shape // 2
    i = np.tile(np.arange(-center_idcs[0], spatial_shape[0]-center_idcs[0])[:, np.newaxis], (1, spatial_shape[1]))
    j = np.tile(np.arange(-center_idcs[1], spatial_shape[1]-center_idcs[1])[np.newaxis, :], (spatial_shape[0], 1))
    return np.linalg.norm(np.concatenate((i, j), axis=3))
