import numpy as np


def soft_color_erosion(multivariate_image, structuring_element, fuzzy_implication_function):
    padded_image = _pad_image_wrt_structuring_element(multivariate_image=multivariate_image,
                                                      structuring_element=structuring_element)
    return _base_soft_color_operator(
        padded_image=padded_image,
        structuring_element=structuring_element,
        se_distances_wrt_center=_euclidean_distance_wrt_center(structuring_element.shape),
        aggregation_function=PrecomputeAggregationFunction(
            first_channel=padded_image[:, :, 0],
            structuring_element=structuring_element,
            aggregation_function=fuzzy_implication_function),
        order_criteria_1=np.min,
        order_criteria_3=np.minimum,
    )


def soft_color_dilation(multivariate_image, structuring_element, fuzzy_conjunction):
    return _base_soft_color_operator(
        padded_image=_pad_image_wrt_structuring_element(multivariate_image=multivariate_image,
                                                        structuring_element=structuring_element),
        structuring_element=structuring_element,
        se_distances_wrt_center=_euclidean_distance_wrt_center(structuring_element.shape),
        aggregation_function=fuzzy_conjunction,
        order_criteria_1=np.max,
        order_criteria_3=np.max,
    )


def _base_soft_color_operator(padded_image, structuring_element, se_distances_wrt_center, aggregation_function,
                              order_criteria_1, order_criteria_3):
    num_channels = padded_image.shape[2]
    sz_se, se_center_idcs, se_before_center, se_after_center = _sizes_wrt_center(structuring_element.shape)
    pad_i, pad_j = _pad_size_wrt_structuring_element(structuring_element=structuring_element)
    sz_result = (padded_image.shape[0] - pad_i[0] - pad_i[1],
                 padded_image.shape[1] - pad_j[0] - pad_j[1])
    se_after_center_included = [e+1 for e in se_after_center]
    eroded_image = np.zeros(shape=sz_result+(num_channels, ))
    for i in range(pad_i[0], padded_image.shape[0] - pad_i[1]):
        im_ini_i = i-se_before_center[0]
        im_end_i = i+se_after_center_included[0]

        for j in range(pad_j[0], padded_image.shape[1] - pad_j[1]):
            im_ini_j = j-se_before_center[1]
            im_end_j = j+se_after_center_included[1]

            im_cropped = padded_image[im_ini_i:im_end_i, im_ini_j:im_end_j, :]
            im_values = im_cropped.reshape((-1, num_channels))
            #  Compute aggregation function
            computed_first_channel = aggregation_function(i, j).ravel()

            # Avoid NANs
            idcs_nonnan = ~np.isnan(computed_first_channel)
            if not np.any(idcs_nonnan):
                eroded_image[i-pad_i[0], j-pad_j[0], :] = np.nan

            else:
                # Order wrt criteria 1 (first-channel value)
                computed_values = np.concatenate((computed_first_channel[idcs_nonnan, np.newaxis], im_values[idcs_nonnan, 1:]),
                                                 axis=1)
                remaining_values = computed_values[:, :]  # Create another view

                optm_value = order_criteria_1(remaining_values[:, 0], axis=0)
                optm_idcs_criteria_1 = np.equal(remaining_values[:, 0], optm_value)
                if optm_idcs_criteria_1.size > 1:
                    # Resolve ties wrt criteria 2 (closest from center)
                    d_center_flattened = se_distances_wrt_center.flatten()

                    d_center_flattened = d_center_flattened[idcs_nonnan]
                    d_center_flattened = d_center_flattened[optm_idcs_criteria_1]

                    optimal_distance = np.min(d_center_flattened, axis=0)
                    optm_idcs_criteria_2 = (d_center_flattened == optimal_distance)
                    if optm_idcs_criteria_2.size != 1:
                        # Resolve ties wrt criteria 3 (lexicographical order)
                        remaining_values = remaining_values[optm_idcs_criteria_1, :]
                        remaining_values = remaining_values[optm_idcs_criteria_2, :]

                        best_idx = 0
                        best_value = remaining_values[best_idx, :]
                        for current_idx in range(1, remaining_values.shape[0]):
                            current_value = remaining_values[current_idx, :]
                            idcs_nonmatching_element = np.where(best_value != current_value)[0]
                            if len(idcs_nonmatching_element) != 0 and order_criteria_3(
                                    current_value[idcs_nonmatching_element[0]],
                                    best_value[idcs_nonmatching_element[0]]):
                                best_idx = current_idx
                                best_value = current_value

                        # Select only best_idx
                        optm_idcs_criteria_2[:] = False
                        optm_idcs_criteria_2[best_idx] = True

                    # Modify only idcs being True (the ones on tie resolution)
                    optm_idcs_criteria_1[optm_idcs_criteria_1] = optm_idcs_criteria_2

                sel_idcs = np.where(optm_idcs_criteria_1)[0]
                eroded_image[i-pad_i[0], j-pad_j[0], :] = computed_values[sel_idcs[0], :]
    return eroded_image


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
                                         self.unique_se_values.size),
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