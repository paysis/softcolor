import numpy as np


def soft_color_erosion(multivariate_image, structuring_element, fuzzy_implication_function):
    eroded_image = np.empty(shape=multivariate_image.shape, dtype=multivariate_image.dtype)
    padded_image = _pad_image_wrt_structuring_element(multivariate_image=multivariate_image,
                                                      structuring_element=structuring_element)
    for limit_i in range(0, multivariate_image.shape[0], 100):
        _base_soft_color_operator(
            padded_image=padded_image,
            structuring_element=structuring_element,
            se_distances_wrt_center=_euclidean_distance_wrt_center(structuring_element.shape),
            aggregation_function=fuzzy_implication_function,
            order_criteria_1=np.min,
            order_criteria_3=np.minimum,
            output=eroded_image,
            range_i=[limit_i, limit_i+100],
            range_j=[0, multivariate_image.shape[1]],
        )
    return eroded_image


def soft_color_dilation(multivariate_image, structuring_element, fuzzy_conjunction):
    return None


def _base_soft_color_operator(padded_image, structuring_element, se_distances_wrt_center, aggregation_function,
                              order_criteria_1, order_criteria_3, output, range_i, range_j):
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
        for idx_j_se in range(sz_se[0]):
            idx_se_flat = np.ravel_multi_index((idx_i_se, idx_j_se), dims=sz_se)
            idx_unique = se_idx_to_unique[idx_se_flat]
            values[:, :, idx_se_flat] = precomputed_unique_se[
                idx_i_se:num_i + idx_i_se,
                idx_j_se:num_j + idx_j_se,
                idx_unique]

    selected_flattened_se_idx = np.nanargmin(values, axis=2)
    grid_val_j, grid_val_i = np.meshgrid(np.arange(values.shape[1]), np.arange(values.shape[0]))
    aggregated_first_channel = values[grid_val_i, grid_val_j, selected_flattened_se_idx]
    idcs_tied_3d = np.equal(values[:, :, :], np.tile(aggregated_first_channel[:, :, np.newaxis], reps=(1, 1, sz_se[0]*sz_se[1])))

    mask_tie = np.sum(idcs_tied_3d, axis=2) == 2
    idx_tie_i, idx_tie_j = np.where(mask_tie)
    for res_i, res_j in zip(idx_tie_i, idx_tie_j):
        pad_ini_i = res_i+range_i[0]+se_before_center[0]-se_before_center[0]
        pad_end_i = res_i+range_i[0]+se_after_center[0]+se_after_center_included[0]
        pad_ini_j = res_j+range_j[0]+se_before_center[1]-se_before_center[1]
        pad_end_j = res_j+range_j[0]+se_after_center[1]+se_after_center_included[1]
        idcs_se_tied = np.where(idcs_tied_3d[res_i, res_j, :])[0]

        compound_data = np.concatenate((+se_distances_wrt_center[:, :, np.newaxis],
                                        padded_image[pad_ini_i:pad_end_i, pad_ini_j:pad_end_j, 1:]),
                                       axis=2)
        compound_data = compound_data.reshape((-1, num_channels))   # num_channels - 1 (first_channel) + 1 (d_se)
        compound_data = compound_data[idcs_se_tied, :]
        best_idx = _lexicographical_argmin(compound_data)
        best_idx = idcs_se_tied[best_idx]
        selected_flattened_se_idx[res_i, res_j] = best_idx

    relative_delta_i, relative_delta_j = np.unravel_index(selected_flattened_se_idx, dims=sz_se)
    grid_out_i = grid_val_i + range_i[0]
    grid_out_j = grid_val_j + range_j[0]
    grid_pad_i = grid_out_i + relative_delta_i
    grid_pad_j = grid_out_j + relative_delta_j

    assert np.all(np.equal(aggregated_first_channel, padded_image[grid_pad_i, grid_pad_j, 0]))
    output[grid_out_i, grid_out_j, 0] = aggregated_first_channel
    for idx_channel in range(1, num_channels):
        output[grid_out_i, grid_out_j, idx_channel] = padded_image[grid_pad_i, grid_pad_j, idx_channel]
    return output


def _euclidean_distance_wrt_center(spatial_shape):
    center_idcs = [e // 2 for e in spatial_shape]
    i = np.tile(np.arange(-center_idcs[0], spatial_shape[0]-center_idcs[0])[:, np.newaxis], (1, spatial_shape[1]))
    j = np.tile(np.arange(-center_idcs[1], spatial_shape[1]-center_idcs[1])[np.newaxis, :], (spatial_shape[0], 1))
    coordinates = np.concatenate((i[:, :, np.newaxis], j[:, :, np.newaxis]), axis=2)
    return np.linalg.norm(coordinates, axis=2)


def _sizes_wrt_center(image_shape):
    img_sz = image_shape[:2]
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
    if np.isnan(min_value):   # TODO: This should not be necessary!
        return 0
    idcs_min = np.where(data[:, 0] == min_value)[0]
    if idcs_min.size == 1:
        return idcs_min[0]
    return idcs_min[_lexicographical_argmin(data[idcs_min, 1:])]


class PrecomputeAggregationFunction:
    def __init__(self, first_channel, structuring_element, aggregation_function):
        self.first_channel = first_channel
        self.img_sh = first_channel.shape
        self.structuring_element = structuring_element.ravel()
        self.aggregation_function = aggregation_function

        sz_se, se_center_idcs, se_before_center, se_after_center = _sizes_wrt_center(structuring_element.shape)
        self.ini_delta = se_before_center
        self.end_delta = [e + 1 for e in se_after_center]

        self.unique_se_values, self.se_to_unique = np.unique(structuring_element, return_inverse=True)

        self.values = None
        if self.first_channel.shape[0] * self.first_channel.shape[1] * self.unique_se_values.size < 1e10:
            # TODO: review that this limit is reasonable.
            self.values = np.full(shape=(self.first_channel.shape[0],
                                         self.first_channel.shape[1],
                                         self.structuring_element.shape[0],
                                         self.structuring_element.shape[1]),
                                  fill_value=np.nan,
                                  dtype='float16')
            self.delta_i = self.end_delta[0] + self.ini_delta[0]
            self.delta_j = self.end_delta[1] + self.ini_delta[1]
            for idx_img_i in range(self.delta_i):
                for idx_img_j in range(self.delta_j):
                    idx_flat = np.ravel_multi_index((idx_img_i, idx_img_j), dims=(self.delta_i, self.delta_j))
                    idx_se_unique = self.se_to_unique[idx_flat]

                    self.values[self.ini_delta[0]:self.end_delta[0],
                                self.ini_delta[1]:self.end_delta[1],
                                idx_se_unique] = self.first_channel[self.ini_delta[0]:self.end_delta[0]]

            unique_se_3D = np.tile(self.unique_se_values[np.newaxis, np.newaxis, :],
                                   reps=(self.first_channel.shape[0], self.first_channel.shape[1], 1))
            with np.warnings.catch_warnings():
                self.values = aggregation_function(unique_se_3D, self.values)



    def __call__(self, img_i_center, img_j_center):
        # Use as a callable emulating aggregation_function(img(i1:i2, j1:j2, 0), se)
        if self.values is None:  # When img/se too big to precompute
            return self._compute_unique_values(img_i_center=img_i_center, img_j_center=img_j_center)

        im_ini_i = img_i_center - self.ini_delta[0]
        im_end_i = img_i_center + self.end_delta[0]
        im_ini_j = img_j_center - self.ini_delta[1]
        im_end_j = img_j_center + self.end_delta[1]
        values_cropped = self.values[im_ini_i:im_end_i, im_ini_j:im_end_j, :]
        return values_cropped[self.idcs_unique_se_3D].reshape(self.delta_i, self.delta_j)

    def _compute_unique_values(self, img_i_center, img_j_center):
        im_ini_i = img_i_center - self.ini_delta[0]
        im_end_i = img_i_center + self.end_delta[0]
        im_ini_j = img_j_center - self.ini_delta[1]
        im_end_j = img_j_center + self.end_delta[1]

        img_cropped = self.first_channel[im_ini_i:im_end_i, im_ini_j:im_end_j].flatten()
        mask_nan_value = np.isnan(img_cropped)
        img_cropped[~mask_nan_value] = self.aggregation_function(
            img_cropped[~mask_nan_value],
            self.structuring_element.flatten()[~mask_nan_value])
        return img_cropped

    def _aggregate_all_combinations(self, flatten_x, flatten_y):
        x = np.tile(flatten_x[:, np.newaxis], reps=(1, flatten_y.size))
        y = np.tile(flatten_y[np.newaxis, :], reps=(flatten_x.size, 1))
        return self.aggregation_function(x, y)