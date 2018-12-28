import math

import numpy as np


def soft_color_erosion(multivariate_image, structuring_element, fuzzy_implication_function):
    return _base_soft_color_operator(multivariate_image=multivariate_image,
                                     structuring_element=structuring_element,
                                     aggregation_function=fuzzy_implication_function,
                                     flag_minimize_first_channel=True,
                                     flag_minimize_euclidean_distance=True,
                                     flag_minimize_lexicographical_order=True)


def soft_color_dilation(multivariate_image, structuring_element, fuzzy_conjunction):
    return _base_soft_color_operator(multivariate_image=multivariate_image,
                                     structuring_element=structuring_element,
                                     aggregation_function=fuzzy_conjunction,
                                     flag_minimize_first_channel=False,
                                     flag_minimize_euclidean_distance=True,
                                     flag_minimize_lexicographical_order=False)


def _base_soft_color_operator(multivariate_image, structuring_element,
                              aggregation_function,
                              flag_minimize_first_channel,
                              flag_minimize_euclidean_distance,
                              flag_minimize_lexicographical_order):
    result_image = np.empty_like(multivariate_image)
    padded_image = _pad_image_wrt_structuring_element(multivariate_image=multivariate_image,
                                                      structuring_element=structuring_element)
    step = _compute_optimum_step(structuring_element)
    for limit_i in range(0, multivariate_image.shape[0], step):
        for limit_j in range(0, multivariate_image.shape[1], step):
                _base_soft_color_operator_limited_range(
                    padded_image=padded_image,
                    structuring_element=structuring_element,
                    se_distances_wrt_center=_euclidean_distance_wrt_center(structuring_element.shape),
                    aggregation_function=aggregation_function,
                    output=result_image,
                    range_i=[limit_i, limit_i+step],
                    range_j=[limit_j, limit_j+step],
                    flag_minimize_first_channel=flag_minimize_first_channel,
                    flag_minimize_euclidean_distance=flag_minimize_euclidean_distance,
                    flag_minimize_lexicographical_order=flag_minimize_lexicographical_order
                )
    return result_image


def _base_soft_color_operator_limited_range(padded_image,
                                            structuring_element, se_distances_wrt_center,
                                            aggregation_function,
                                            output, range_i, range_j,
                                            flag_minimize_first_channel,
                                            flag_minimize_euclidean_distance,
                                            flag_minimize_lexicographical_order):
    num_channels = padded_image.shape[2]
    sz_se, se_center_idcs, se_before_center, se_after_center = _sizes_wrt_center(structuring_element.shape)
    pad_i, pad_j = _pad_size_wrt_structuring_element(structuring_element=structuring_element)
    se_after_center_included = [e+1 for e in se_after_center]

    range_i[1] = min(range_i[1], output.shape[0])
    range_j[1] = min(range_j[1], output.shape[0])
    num_i = range_i[1] - range_i[0]
    num_j = range_j[1] - range_j[0]

    # Precompute AggregationFunction(ImageWithOffset, SE_uniqueValues)
    se_uniques, se_unique_to_idx, se_idx_to_unique = np.unique(structuring_element,
                                                               return_index=True, return_inverse=True)
    precomputed_unique_se = np.empty(shape=(num_i+pad_i[0]+pad_i[1],
                                            num_j+pad_j[0]+pad_j[1],
                                            se_uniques.size),
                                     dtype=output.dtype)
    for idx_unique in range(se_uniques.size):
        idx_se_flat = se_unique_to_idx[idx_unique]
        idx_i_se, idx_j_se = np.unravel_index(idx_se_flat, dims=sz_se)
        if np.isnan(structuring_element[idx_i_se, idx_j_se]):
            precomputed_unique_se[:, :, idx_unique] = np.nan
        else:
            cropped_first_channel = padded_image[
                range_i[0]:range_i[1]+se_before_center[0]+se_after_center[0],
                range_j[0]:range_j[1]+se_before_center[0]+se_after_center[1],
                0].copy()
            mask_nans = np.isnan(cropped_first_channel)
            cropped_first_channel[~mask_nans] = aggregation_function(
                np.full(shape=(np.count_nonzero(~mask_nans), ),
                        fill_value=structuring_element[idx_i_se, idx_j_se],
                        dtype=structuring_element.dtype),
                cropped_first_channel[~mask_nans],
            )
            precomputed_unique_se[:, :, idx_unique] = cropped_first_channel

    values = np.empty(shape=(num_i, num_j, sz_se[0] * sz_se[1]), dtype=output.dtype)
    for idx_i_se in range(sz_se[0]):
        for idx_j_se in range(sz_se[1]):
            idx_se_flat = np.ravel_multi_index((idx_i_se, idx_j_se), dims=sz_se)
            idx_unique = se_idx_to_unique[idx_se_flat]
            values[:, :, idx_se_flat] = precomputed_unique_se[
                idx_i_se:num_i + idx_i_se,
                idx_j_se:num_j + idx_j_se,
                idx_unique]
    values_allnan_mask = np.all(np.isnan(values), axis=2)
    if np.any(values_allnan_mask):
        values_flat = values.reshape((-1, 1, values.shape[2]))
        mask_flat = values_allnan_mask.reshape((-1, ))
        idcs_flat = np.zeros(shape=values_flat.shape[:2], dtype='uint64')
        if flag_minimize_first_channel:
            idcs_flat[~mask_flat, :] = np.nanargmin(values_flat[~mask_flat, :, :], axis=2)
        else:
            idcs_flat[~mask_flat, :] = np.nanargmax(values_flat[~mask_flat, :, :], axis=2)
        selected_se_idx = idcs_flat.reshape(num_i, num_j)

    else:
        if flag_minimize_first_channel:
            selected_se_idx = np.nanargmin(values, axis=2)
        else:
            selected_se_idx = np.nanargmax(values, axis=2)
    grid_val_j, grid_val_i = np.meshgrid(np.arange(values.shape[1]), np.arange(values.shape[0]))
    aggregated_first_channel = values[grid_val_i, grid_val_j, selected_se_idx]
    idcs_tied_3d = np.equal(values[:, :, :], aggregated_first_channel[:, :, np.newaxis])

    mask_tie = np.sum(idcs_tied_3d, axis=2) != 1
    idx_tie_i, idx_tie_j = np.where(mask_tie & ~values_allnan_mask)
    for res_i, res_j in zip(idx_tie_i, idx_tie_j):
        pad_ini_i = res_i+range_i[0]+se_before_center[0]-se_before_center[0]
        pad_end_i = res_i+range_i[0]+se_after_center[0]+se_after_center_included[0]
        pad_ini_j = res_j+range_j[0]+se_before_center[1]-se_before_center[1]
        pad_end_j = res_j+range_j[0]+se_after_center[1]+se_after_center_included[1]
        idcs_se_tied = np.where(idcs_tied_3d[res_i, res_j, :])[0]

        if flag_minimize_euclidean_distance != flag_minimize_lexicographical_order:
            sign_d = 1
        else:
            sign_d = -1
        compound_data = np.concatenate((sign_d * se_distances_wrt_center[:, :, np.newaxis],
                                        padded_image[pad_ini_i:pad_end_i, pad_ini_j:pad_end_j, 1:]),
                                       axis=2)
        compound_data = compound_data.reshape((-1, num_channels))   # num_channels - 1 (first_channel) + 1 (d_se)
        compound_data = compound_data[idcs_se_tied, :]
        if flag_minimize_lexicographical_order:
            best_idx = _lexicographical_argmin(compound_data)
        else:
            best_idx = _lexicographical_argmax(compound_data)
        best_idx = idcs_se_tied[best_idx]
        selected_se_idx[res_i, res_j] = best_idx

    relative_delta_i, relative_delta_j = np.unravel_index(selected_se_idx, dims=sz_se)
    grid_out_i = grid_val_i + range_i[0]
    grid_out_j = grid_val_j + range_j[0]
    grid_pad_i = grid_out_i + relative_delta_i
    grid_pad_j = grid_out_j + relative_delta_j

    aggregated_first_channel[values_allnan_mask] = np.nan
    output[grid_out_i, grid_out_j, 0] = aggregated_first_channel
    for idx_channel in range(1, num_channels):
        channel = padded_image[grid_pad_i, grid_pad_j, idx_channel]
        channel[values_allnan_mask] = np.nan
        output[grid_out_i, grid_out_j, idx_channel] = channel
    return output


def _euclidean_distance_wrt_center(spatial_shape):
    # Return array containing its distance to the center.
    center_idcs = [e // 2 for e in spatial_shape]
    i = np.tile(np.arange(-center_idcs[0], spatial_shape[0]-center_idcs[0])[:, np.newaxis], (1, spatial_shape[1]))
    j = np.tile(np.arange(-center_idcs[1], spatial_shape[1]-center_idcs[1])[np.newaxis, :], (spatial_shape[0], 1))
    coordinates = np.concatenate((i[:, :, np.newaxis], j[:, :, np.newaxis]), axis=2)
    return np.linalg.norm(coordinates, axis=2)


def _sizes_wrt_center(spatial_shape):
    # Measure lengths wrt the center of the image (i.e. before/after center).
    img_sz = spatial_shape[:2]
    center_idcs = [e//2 for e in img_sz]
    sz_before_center_excluded = center_idcs
    after_center_included = [e1 - e2 for e1, e2 in zip(img_sz, center_idcs)]
    sz_after_center_excluded = [e-1 for e in after_center_included]
    return img_sz, center_idcs, sz_before_center_excluded, sz_after_center_excluded


def _pad_size_wrt_structuring_element(structuring_element):
    _, _, se_before_center, se_after_center = _sizes_wrt_center(structuring_element.shape)
    return (se_before_center[0], se_after_center[0]), (se_before_center[1], se_after_center[1])


def _pad_image_wrt_structuring_element(multivariate_image, structuring_element):
    pad_i, pad_j = _pad_size_wrt_structuring_element(structuring_element=structuring_element)
    padded_image = np.pad(multivariate_image,
                          (pad_i, pad_j, (0, 0)),
                          'constant', constant_values=(np.nan, np.nan))
    return padded_image


def _lexicographical_argmin(data):
    # Data must be two-dimensional, being the first axis the one to be sorted
    # Returns a numeric index
    if data.shape[1] == 1:
        return np.argmin(data[:, 0])
    min_value = np.nanmin(data[:, 0])
    idcs_min = np.where(data[:, 0] == min_value)[0]
    if idcs_min.size == 1:
        return idcs_min[0]
    return idcs_min[_lexicographical_argmin(data[idcs_min, 1:])]


def _lexicographical_argmax(data):
    # Data must be two-dimensional, being the first axis the one to be sorted
    # Returns a numeric index
    if data.shape[1] == 1:
        return np.argmax(data[:, 0])
    max_value = np.nanmax(data[:, 0])
    idcs_max = np.where(data[:, 0] == max_value)[0]
    if idcs_max.size == 1:
        return idcs_max[0]
    return idcs_max[_lexicographical_argmin(data[idcs_max, 1:])]


def _compute_optimum_step(structuring_element):
    # Returns appropriate value for batch size B x B (we create potentially large arrays B x B x SE.size).
    max_array_size = 3e8
    step = math.sqrt(max_array_size / structuring_element.size)
    step = math.ceil(step)
    step = max(step, structuring_element.size)  # To avoid redundant computations due to padded sub-images
    return step
