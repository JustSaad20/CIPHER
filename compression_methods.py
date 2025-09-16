import numpy as np
import matplotlib.pyplot as pltw
from fpc import *
from huffman import *

def weight_8_to_32(quant_weight_in):
    quant_weight_in = quant_weight_in.astype('uint8').reshape(-1)
    if len(quant_weight_in) % 4 != 0:
        padding_needed = 4 - (len(quant_weight_in) % 4)
        quant_weight_in = np.concatenate([np.zeros(padding_needed, dtype=np.uint8), quant_weight_in])
    else:
        padding_needed = 0

    weight_in32 = ((quant_weight_in[ ::4].astype(np.uint32) << 24) |
                   (quant_weight_in[1::4].astype(np.uint32) << 16) |
                   (quant_weight_in[2::4].astype(np.uint32) << 8)  |
                   (quant_weight_in[3::4]))
    return weight_in32, padding_needed

def err_handler_8(weight_in, errlvl1, errlvl2, save_state):
    weight_in_8 = weight_in.astype('uint8').reshape(-1)
    weight_in_32, padding_needed = weight_8_to_32(weight_in_8)

    unpacked = np.unpackbits(weight_in_32.byteswap(inplace=True).view(np.uint8))
    unpacked = unpacked.reshape(-1, 32)

    return_array = np.apply_along_axis(fpc_compress_word, 1, unpacked)
    key = return_array[:, :2]
    unfilt_word = return_array[:, 2:]
    word = unfilt_word[unfilt_word != -1]
    word_inject = word.reshape(-1, 2)
    if save_state == 0:
        prccsd_weight = error_injection(word_inject, errlvl1, errlvl2)
    elif save_state == 1:
        prccsd_weight = error_injection_with_SLC(word_inject, errlvl1, errlvl2)
    else:
        prccsd_weight = error_injection_with_SMARTSLC(word_inject, errlvl1, errlvl2)

    reconstructed_weights = bit_reconstructor_32(key, prccsd_weight.flatten())
    reconstructed_weights = reconstructed_weights[ padding_needed :]

    return reconstructed_weights


def fpc_protocol_reduced(quant_weight_in, errlvl1, errlvl2, save_state, stat_print=0):
    weight_in32, padding_needed = weight_8_to_32(quant_weight_in)

    unpacked_word = np.unpackbits(weight_in32.byteswap(inplace=True).view(np.uint8))
    reshaped_unpacked_word = unpacked_word.reshape(-1, 32)

    return_array = np.apply_along_axis(fpc_compress_word_reduced, 1, reshaped_unpacked_word)
    key = return_array[:, :2]
    unfilt_word = return_array[:, 2:]
    word = unfilt_word[unfilt_word != -1]
    word_inject = word.reshape(-1, 2)
    
    if stat_print == 0:
        if save_state == 0:
            prccsd_weight = error_injection(word_inject, errlvl1, errlvl2)
        elif save_state == 1:
            prccsd_weight = error_injection_with_SLC(word_inject, errlvl1, errlvl2)
        else:
            prccsd_weight = error_injection_with_SMARTSLC(word_inject, errlvl1, errlvl2)

        reconstructed_weights = bit_reconstructor_32_reduced(key, prccsd_weight.flatten())
        reconstructed_weights = reconstructed_weights[ padding_needed :]
        return reconstructed_weights
    else:
        fpc_compression_stats(unpacked_word.flatten(), word.flatten(), key.flatten())

        
def fpc_protocol(quant_weight_in, errlvl1, errlvl2, save_state, stat_print=0):
    weight_in32, padding_needed = weight_8_to_32(quant_weight_in)

    unpacked_word = np.unpackbits(weight_in32.byteswap(inplace=True).view(np.uint8))
    reshaped_unpacked_word = unpacked_word.reshape(-1, 32)

    return_array = np.apply_along_axis(fpc_compress_word, 1, reshaped_unpacked_word)
    key = return_array[:, :3]
    unfilt_word = return_array[:, 3:]
    word = unfilt_word[unfilt_word != -1]
    word_inject = word.reshape(-1, 2)
    if stat_print == 0:
        if save_state == 0:
            prccsd_weight = error_injection(word_inject, errlvl1, errlvl2)
        elif save_state == 1:
            prccsd_weight = error_injection_with_SLC(word_inject, errlvl1, errlvl2)
        else:
            prccsd_weight = error_injection_with_SMARTSLC(word_inject, errlvl1, errlvl2)

        reconstructed_weights = bit_reconstructor_32(key, prccsd_weight.flatten())
        reconstructed_weights = reconstructed_weights[ padding_needed :]
        return reconstructed_weights
    else:
        fpc_compression_stats(unpacked_word.flatten(), word.flatten(), key.flatten())

def huffman_protocol(data_in, size =0):
    huffman_tree = build_huffman_tree(data_in)
    huffman_codes = build_huffman_codes(huffman_tree)
    encoded_data = encode(data_in, huffman_codes)
    decoded_data = decode(encoded_data, huffman_tree)
    if (decoded_data == data_in).all():
        print("True")
    else:
        print("False")

    # tree_size_bits = calculate_tree_size(huffman_tree, size_per_symbol=size, size_per_freq=32)
    # huffman_compression_stats(data_in, encoded_data, size, tree_size_bits)


def fpc_compression_stats(original_data, encoded_data, encoded_key):
    original_size = len(original_data)
    compressed_size = len(encoded_data) +len(encoded_key)
    print(f"Original size: {original_size} bits")
    print(f"Compressed size (including key): {compressed_size} bits")
    print(f"Compression ratio: {compressed_size / original_size:.2%}")

def huffman_compression_stats(original_data, encoded_data, size, tree_size_bits):
    original_size = len(original_data) * size  # size in bits (1 byte = 8 bits)
    compressed_size = len(encoded_data) + tree_size_bits  # Add Huffman tree size
    print(f"Original size: {original_size} bits")
    print(f"Compressed size (including tree): {compressed_size} bits")
    print(f"Compression ratio: {compressed_size / original_size:.2%}")