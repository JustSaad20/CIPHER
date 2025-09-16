from error_injection import *

def fpc_compress_word(input_word):
    if np.sum(input_word) == 0:
        compressed_key = np.array([0, 0, 0])
        compressed_word = np.array([])
    elif ((np.sum(input_word[0:(28+1)]) == 0) or (np.sum(input_word[0:(28+1)]) == 28+1)):
        compressed_key = np.array([0, 0, 1])
        compressed_word = input_word[28:32]
    elif ((np.sum(input_word[0:(24+1)]) == 0) or (np.sum(input_word[0:(24+1)]) == 24+1)):
        compressed_key = np.array([0, 1, 0])
        compressed_word = input_word[24:32]
    elif ((np.sum(input_word[0:(16+1)]) == 0) or (np.sum(input_word[0:(16+1)]) == 16+1)):
        compressed_key = np.array([0, 1, 1])
        compressed_word = input_word[16:32]
    elif ((np.sum(input_word[0:(16)]) == 0)):
        compressed_key = np.array([1, 0, 0])
        compressed_word = input_word[16:32]
    elif (((np.sum(input_word[0:(8+1)]) == 0) or (np.sum(input_word[0:(8+1)]) == 8+1)) and ((np.sum(input_word[16:(24+1)]) == 0) or (np.sum(input_word[16:(24+1)]) == 8+1))):
        compressed_key = np.array([1, 0, 1])
        compressed_word = np.concatenate([input_word[8:16], input_word[24:32]])
    elif (np.array_equal(input_word[0:8], input_word[8:16]) and np.array_equal(input_word[8:16], input_word[16:24]) and np.array_equal(input_word[16:24], input_word[24:32])):
        compressed_key = np.array([1, 1, 0])
        compressed_word = input_word[0:8]
    else:
        compressed_key = np.array([1, 1, 1])
        compressed_word = input_word
    return_word = np.concatenate((compressed_key, np.full((32 - compressed_word.size), -1), compressed_word)).astype(int)#padded with -1 to ensure np.apply along axis
    return return_word

def fpc_compress_word_reduced(input_word):
    if ((np.sum(input_word[0:(31+1)]) == 0) or (np.sum(input_word[0:(31+1)]) == 31+1)):
        compressed_key = np.array([0, 0])
        compressed_word = input_word[31:32]
    elif ((np.sum(input_word[0:(28+1)]) == 0) or (np.sum(input_word[0:(28+1)]) == 28+1)):
        compressed_key = np.array([0, 1])
        compressed_word = input_word[28:32]
    elif ((np.sum(input_word[0:(24+1)]) == 0) or (np.sum(input_word[0:(24+1)]) == 24+1)):
        compressed_key = np.array([1, 0])
        compressed_word = input_word[24:32]
    else:
        compressed_key = np.array([1, 1])
        compressed_word = input_word

    return_word = np.concatenate((compressed_key, np.full((32 - compressed_word.size), -1), compressed_word)).astype(int) #padded with -1 so that apply_along axis can be applied 
    return return_word

def bit_reconstructor_32(keys, compressed_word):
    original_data = np.empty((keys.shape[0], 32), dtype=np.uint8)
    word_index = 0

    for i in range(keys.shape[0]):
        if np.array_equal(keys[i], [0, 0, 0]):
            original_data[i] = 0
        elif np.array_equal(keys[i], [0, 0, 1]):
            original_data[i] = np.concatenate((np.tile(compressed_word[word_index],28),compressed_word[word_index:word_index + 4]))
            word_index += 4
        elif np.array_equal(keys[i], [0, 1, 0]):
            original_data[i] = np.concatenate((np.tile(compressed_word[word_index],24),compressed_word[word_index:word_index + 8]))
            word_index += 8
        elif np.array_equal(keys[i], [0, 1, 1]):
            original_data[i] = np.concatenate((np.tile(compressed_word[word_index],16),compressed_word[word_index:word_index + 16]))
            word_index += 16
        elif np.array_equal(keys[i], [1, 0, 0]):
            original_data[i] = np.concatenate((np.tile([0], 16),compressed_word[word_index:word_index + 16]))
            word_index += 16
        elif np.array_equal(keys[i], [1, 0, 1]):
            original_data[i] = original_data[i] = np.concatenate((np.tile(compressed_word[word_index], 8), compressed_word[word_index:word_index + 8], np.tile(compressed_word[word_index + 8], 8), compressed_word[word_index + 8:word_index + 16]))
            word_index += 16
        elif np.array_equal(keys[i], [1, 1, 0]):
            original_data[i] = np.tile(compressed_word[word_index:word_index + 8],4)
            word_index += 8
        else:
            original_data[i] = compressed_word[word_index:word_index + 32]
            word_index += 32

    original_data = (np.packbits(original_data.reshape(-1, 8))).astype('int8')
    return original_data

def bit_reconstructor_32_reduced(keys, compressed_word):
    original_data = np.empty((keys.shape[0], 32), dtype=np.uint8)
    word_index = 0

    for i in range(keys.shape[0]):
        if np.array_equal(keys[i], [0, 0]):
            original_data[i] = np.concatenate((np.tile(compressed_word[word_index],31),compressed_word[word_index:word_index + 1]))
            word_index += 1
        elif np.array_equal(keys[i], [0, 1]):
            original_data[i] = np.concatenate((np.tile(compressed_word[word_index],28),compressed_word[word_index:word_index + 4]))
            word_index += 4
        elif np.array_equal(keys[i], [1, 0]):
            original_data[i] = np.concatenate((np.tile(compressed_word[word_index],24),compressed_word[word_index:word_index + 8]))
            word_index += 8
        else:
            original_data[i] = compressed_word[word_index:word_index + 32]
            word_index += 32

    original_data = (np.packbits(original_data.reshape(-1, 8))).astype('int8')
    return original_data



    