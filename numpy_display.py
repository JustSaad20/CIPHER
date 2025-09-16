from fpc import *
from huffman import *
from methods import *
import glob

resnet_path = glob.glob("NumpyWeights/Quantized_Weights_16*weight_resnet18_.npy")
lenet_path = glob.glob("NumpyWeights/Quantized_Weights_16*weight_LeNet_.npy")
inception_path = glob.glob("NumpyWeights/Quantized_Weights_16*weight_Inception_.npy")



for file_path in resnet_path:
    resnet_data = np.load(file_path).astype(np.uint8)
for file_path in lenet_path:
    lenet_data = np.load(file_path).astype(np.uint8)
for file_path in inception_path:
    inception_data = np.load(file_path).astype(np.uint8)

# quantized 8 bits to 32
resnet_data32,throw = weight_8_to_32(resnet_data)
lenet_data32,throw = weight_8_to_32(lenet_data)
inception_data32,throw = weight_8_to_32(inception_data)

#  TODO: decide plot frequencies for display
plot_frequency(resnet_data,"resnet_data.png")
plot_frequency(resnet_data32,"resnet_data32.png")
plot_frequency(lenet_data,"lenet_data.png")
plot_frequency(lenet_data32,"lenet_data32.png")
plot_frequency(inception_data,"inception_data.png")
plot_frequency(inception_data32,"inception_data32.png")


huffman_protocol(resnet_data, 16)
huffman_protocol(resnet_data32, 64)
huffman_protocol(lenet_data, 16)
huffman_protocol(lenet_data32, 64)
huffman_protocol(inception_data, 16)
huffman_protocol(inception_data32, 64)

    
# fpc_protocol(resnet_data, 0, 0, 0, 1)
# fpc_protocol_reduced(resnet_data, 0, 0, 0, 1)

# fpc_protocol(lenet_data, 0, 0, 0, 1)
# fpc_protocol_reduced(lenet_data, 0, 0, 0, 1)

# fpc_protocol(inception_data, 0, 0, 0, 1)
# fpc_protocol_reduced(inception_data, 0, 0, 0, 1)




#{original size, compression ratio for 3x4}, frequency of data occurance
# data_32 = weight_8_to_32(data)
# huffman_protocol(data_32, 32)