def error_injection(prcssd_weight, mlc_error_lvl1, mlc_error_lvl2):
    unique, counts = np.unique(prcssd_weight, axis=0, return_counts=True)
    # Get indices that would sort the counts array in ascending order
    sorted_indices = np.argsort(counts)
    # Sort the unique elements according to the sorted_indices
    sorted_unique = unique[sorted_indices]

    # Generate a random matrix with the same shape as prcssd_weight
    random_matrix = np.random.rand(*prcssd_weight.shape)

    # Create a mask for sub-arrays to find specified value
    mask_lvl1 = np.all(prcssd_weight == sorted_unique[1] , axis=1)
    random_mask_lvl1 = random_matrix[:, 0] < mlc_error_lvl1

    mask_lvl2 = np.all(prcssd_weight == sorted_unique[0] , axis=1)
    random_mask_lvl2 = random_matrix[:, 0] < mlc_error_lvl2

    # Combine masks to find where both conditions are met
    flip_mask_lvl1 = np.logical_and(mask_lvl1, random_mask_lvl1)
    flip_mask_lvl2 = np.logical_and(mask_lvl2, random_mask_lvl2)


    # Use the mask to flip sub-arrays
    prcssd_weight[flip_mask_lvl1] = sorted_unique[0]
    prcssd_weight[flip_mask_lvl2] = sorted_unique[3]
    return prcssd_weight

def error_injection_with_SLC(prcssd_weight, mlc_error_lvl1, mlc_error_lvl2):
    unique, counts = np.unique(prcssd_weight, axis=0, return_counts=True)
    # Get indices that would sort the counts array in ascending order
    sorted_indices = np.argsort(counts)
    # Sort the unique elements according to the sorted_indices
    sorted_unique = unique[sorted_indices]

    # Generate a random matrix with the same shape as prcssd_weight
    random_matrix = np.random.rand(*prcssd_weight.shape)

    # Create a mask for sub-arrays to find specified value
    mask_lvl1 = np.all(prcssd_weight == sorted_unique[1], axis=1)
    random_mask_lvl1 = random_matrix[:, 0] < mlc_error_lvl1

    mask_lvl2 = np.all(prcssd_weight == sorted_unique[0], axis=1)
    random_mask_lvl2 = random_matrix[:, 0] < mlc_error_lvl2

    # Combine masks to find where both conditions are met
    flip_mask_lvl1 = np.logical_and(mask_lvl1, random_mask_lvl1)
    flip_mask_lvl2 = np.logical_and(mask_lvl2, random_mask_lvl2)

    even_bits = np.arange(prcssd_weight.shape[0]) % 2 == 0
    flip_mask_lvl1 &= even_bits
    flip_mask_lvl2 &= even_bits

    # Use the mask to flip sub-arrays
    prcssd_weight[flip_mask_lvl1] = sorted_unique[0]
    prcssd_weight[flip_mask_lvl2] = sorted_unique[3]

    return prcssd_weight

def SMART_SLC_encoding(weight, sorted_unique):
    weight = weight.reshape(-1,2)
    if np.all(weight[0] == sorted_unique[3]) or np.all(weight[0] == sorted_unique[2]):
        return [True, False]
    else:
        return [False, True]

def error_injection_with_SMARTSLC(prcssd_weight, mlc_error_lvl1, mlc_error_lvl2):
    unique, counts = np.unique(prcssd_weight, axis=0, return_counts=True)
    # Get indices that would sort the counts array in ascending order
    sorted_indices = np.argsort(counts)
    # Sort the unique elements according to the sorted_indices
    sorted_unique = unique[sorted_indices]

    # Reshape prcssd_weight for proper application of SMART_SLC_encoding
    weight_check = prcssd_weight.copy().reshape(-1, 4)

    # Apply SMART_SLC_encoding to each row
    SMART_Mask = np.apply_along_axis(SMART_SLC_encoding, axis=1, arr=weight_check, sorted_unique=sorted_unique)

    # Flatten the resulting SMART_Mask
    SMART_Mask = SMART_Mask.flatten()
    # Generate a random matrix with the same shape as prcssd_weight
    random_matrix = np.random.rand(*prcssd_weight.shape)

    # Create a mask for sub-arrays to find specified value
    mask_lvl1 = np.all(prcssd_weight == sorted_unique[1], axis=1)
    random_mask_lvl1 = random_matrix[:, 0] < mlc_error_lvl1

    mask_lvl2 = np.all(prcssd_weight == sorted_unique[0], axis=1)
    random_mask_lvl2 = random_matrix[:, 0] < mlc_error_lvl2

    # Combine masks to find where both conditions are met
    flip_mask_lvl1 = np.logical_and(mask_lvl1, random_mask_lvl1)
    flip_mask_lvl2 = np.logical_and(mask_lvl2, random_mask_lvl2)
    # Apply SMART_Mask to flip_mask_lvl1 and flip_mask_lvl2
    flip_mask_lvl1 &= SMART_Mask
    flip_mask_lvl2 &= SMART_Mask

    # Use the mask to flip sub-arrays
    prcssd_weight[flip_mask_lvl1] = sorted_unique[0]
    prcssd_weight[flip_mask_lvl2] = sorted_unique[3]
    return prcssd_weight