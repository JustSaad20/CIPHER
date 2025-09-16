from collections import Counter
import numpy as np
import heapq

class HuffmanNode:
    def __init__(self, symbol=None, freq=0):
        self.symbol = symbol
        self.freq = freq
        self.left = None
        self.right = None

    def __lt__(self, other):
        return self.freq < other.freq

def build_huffman_tree(data):
    frequency = Counter(data)
    priority_queue = [HuffmanNode(symbol=sym, freq=freq) for sym, freq in frequency.items()]
    heapq.heapify(priority_queue)

    while len(priority_queue) > 1:
        left = heapq.heappop(priority_queue)
        right = heapq.heappop(priority_queue)
        merged = HuffmanNode(freq=left.freq + right.freq)
        merged.left = left
        merged.right = right
        heapq.heappush(priority_queue, merged)

    return priority_queue[0]

def calculate_tree_size(node, size_per_symbol=8, size_per_freq=32):
    if node is None:
        return 0

    # Size of the current node
    node_size = size_per_freq
    if node.symbol is not None:  # Leaf node stores a symbol
        node_size += size_per_symbol

    # Recursively calculate size for left and right subtrees
    left_size = calculate_tree_size(node.left, size_per_symbol, size_per_freq)
    right_size = calculate_tree_size(node.right, size_per_symbol, size_per_freq)

    return node_size + left_size + right_size

def build_huffman_codes(node, code="", codebook=None):
    if codebook is None:
        codebook = {}

    if node.symbol is not None:  #leaf node
        codebook[node.symbol] = code
        return codebook

    if node.left:
        build_huffman_codes(node.left, code + "0", codebook)
    if node.right:
        build_huffman_codes(node.right, code + "1", codebook)

    return codebook

def encode(data, codebook):
    return "".join(codebook[sym] for sym in data)


def decode(encoded_data, huffman_tree):
    decoded_output = []
    current_node = huffman_tree

    for bit in encoded_data:
        if bit == '0':
            current_node = current_node.left
        else:  # bit == '1'
            current_node = current_node.right

        if current_node.symbol is not None:  # Leaf node
            decoded_output.append(current_node.symbol)
            current_node = huffman_tree

    return np.array(decoded_output, dtype=type(decoded_output[0]))
