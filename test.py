from compression_methods import *

# example = np.arange(0, 13, dtype=np.int8)
example = np.load("weights_import/example.npy").astype(np.uint8)

huffman_protocol(example)
fpc_protocol(example, 0, 0, 0, 1)
fpc_protocol_reduced(example, 0, 0, 0, 1)


if (fpc_protocol == example).all():
    print("True")
else:
    print("False")


if (fpc_protocol_reduced == example).all():
    print("True")
else:
    print("False")