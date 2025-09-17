"""Test quantized model"""
from cifar10_models import *
from functools import partial
from compression_methods import *
from torch.utils.tensorboard import SummaryWriter
from torchsummary import summary
from tqdm import tqdm
from utils import progress_bar
from compression_methods import *
import argparse
import copy
import logging
import math
import numpy as np
import os
import pandas as pd
import random
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.quantization
import torchvision
import torchvision.transforms as transforms
import traceback
import sys
import matplotlib.pyplot as plt
import pickle



torch.set_printoptions(profile="full")
np.set_printoptions(threshold=sys.maxsize)

msglogger = logging.getLogger()
parser = argparse.ArgumentParser(description="PyTorch CIFAR10 Training")
parser.add_argument("--lr", default=0.1, type=float, help="learning rate")
parser.add_argument("--mlc", default=8, type=int, help="Number of mlc bits")
parser.add_argument("--name", default="Name", type=str, help="Name of run")
parser.add_argument("--model", default="resnet18", type=str, help="Model")
parser.add_argument("--gpu", default="0", type=str, help="GPU ids")
parser.add_argument("--error_pat", default="00", type=str, help="error pattern")
parser.add_argument("--des_pat", default="00", type=str, help="destination pattern")
parser.add_argument("--save_data", "-s", action="store_true", help="Save the data")
parser.add_argument("--encode", "-e", action="store_true", help="Enable encode for flipcy and helmet")
parser.add_argument("--resume", "-r", action="store_true", help="resume from checkpoint")
parser.add_argument("--num_bits", default=8, type=int, help="Number of quantized bits")
# parser.add_argument("--save", default=0, type=bool, help="whether to save as image or not")

args = parser.parse_args()

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = f"{args.gpu}"


def main():
    # savename = f"Results/{args.model}_{args.num_bits}_{args.method}_{args.numberofcentroids}_{args.injectionmode}_{args.duplicate}.png"
    # if args.save == 1 and os.path.exists(savename):
    #     print("already saved")
    # else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        best_acc = 0  # best test accuracy
        start_epoch = 0  # start from epoch 0 or last checkpoint epoch
        # Data
        print("==> Preparing data..")
        transform_train = transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ]
        )

        transform_test = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ]
        )

        trainset = torchvision.datasets.CIFAR10(
            root="./Datasets/cifar10/", train=True, download=False, transform=transform_train
        )
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=500, shuffle=True, num_workers=2)

        testset = torchvision.datasets.CIFAR10(
            root="./Datasets/cifar10/", train=False, download=False, transform=transform_test
        )
        _, val_set = torch.utils.data.random_split(testset, [9500, 500])

        testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)
        valloader = torch.utils.data.DataLoader(val_set, batch_size=100, shuffle=False, num_workers=2)

        classes = (
            "plane",
            "car",
            "bird",
            "cat",
            "deer",
            "dog",
            "frog",
            "horse",
            "ship",
            "truck",
        )

        # Model
        print("==> Building model..")
        if args.model == "resnet18":
            net = resnet18()
            net.load_state_dict(torch.load("./checkpoint/resnet18.pt"))
        elif args.model == "LeNet":
            net = googlenet()
            net.load_state_dict(torch.load("./checkpoint/googlenet.pt"))
        elif args.model == "Inception":
            net = Inception3()
            net.load_state_dict(torch.load("./checkpoint/inception_v3.pt"))
        elif args.model == "vgg16":
            net = vgg16_bn()
            net.load_state_dict(torch.load("./checkpoint/vgg16_bn.pt"))

        net = net.to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)

        orig_state_dict = copy.deepcopy(net.state_dict())

        iteration = 100
        tensorboard = args.save_data
        save_data = args.save_data
        if save_data:
            df = pd.DataFrame(columns=["Time", "Acc."])
        if tensorboard:
            ran = random.randint(1, 100)
            writer = SummaryWriter(f"./runs/{args.name}-{ran}")
            print(f"Run ID: {ran}")

        if args.num_bits == 8:
            dtype = np.int8
        else:
            dtype = np.int16

            states = 3

        with torch.no_grad():
            loop_time = 1
            accuraciesMLC = []
            accuraciesSLC = []
            accuraciesSMARTSLC = []
            csvloader = pd.read_csv("./injected_error_levels/error_level2_1year.csv", header=None, delimiter="\t")
            error_lvl2 = csvloader.values.reshape(-1)
            csvloader = pd.read_csv("./injected_error_levels/error_level3_1year.csv", header=None, delimiter="\t")
            error_lvl3 = csvloader.values.reshape(-1)
            for error_values in range(25):
                for save_state in range(2,3):
                  total_acc = 0
                  print(f"Running with error rate of mlc lvl2: {error_lvl2[error_values]}, and error rate of mlc lvl3: {error_lvl3[error_values]}")
                  for loop in range(loop_time):
                      net.load_state_dict(orig_state_dict)                      # Load the original weights at the beginning of each loop
                      for name, weight in net.named_parameters():
                          if ("weight" in name) and ("bn" not in name):
                            #Dynamic fixed-point quantization
                            sf = args.num_bits - 1. - compute_integral_part(weight, overflow_rate=0.0)
                            quantized_weight, delta = linear_quantize(weight, sf, bits=args.num_bits)
                            shape = weight.shape
                            quantized_weight = quantized_weight.view(-1).detach().cpu().numpy()
                            quantized_weight = quantized_weight.astype(dtype)

                            # filename = f"weights_import/Quantized_Weights_{args.num_bits}_{name}_{args.model}_.npy"
                            # if not os.path.exists(filename):
                            #     np.save(filename, quantized_weight)

                            encoded_weight = fpc_protocol(quantized_weight, error_lvl2[error_values], error_lvl3[error_values], 0)

                            # Decoding and fault inject
                            quantized_weight_torch = torch.from_numpy(encoded_weight.astype(float))
                            quantized_weight_torch = quantized_weight_torch.reshape(shape).cuda()

                            # Dequantization:
                            dequantized_weight = quantized_weight_torch * delta
                            weight.copy_(dequantized_weight)

                      acc = test(net, criterion, optimizer, testloader, device)
                      total_acc = total_acc + acc
                  if save_state == 0:
                    accuraciesMLC.append(total_acc / loop_time)
                  elif save_state == 1:
                    accuraciesSLC.append(total_acc / loop_time)
                  else:
                    accuraciesSMARTSLC.append(total_acc / loop_time)

            result_data = pd.DataFrame({
                'AccuraciesSMARTSLC': accuraciesSMARTSLC,
            })
            savename = f"Results/{args.model}_SMARTSLC.csv"
            result_data.to_csv(savename, index=False)
            
            # flip_chances = np.logspace(1, 25, num=25, base=2)
            # result_data = pd.DataFrame({
            #     'AccuraciesMLC': accuraciesMLC,
            #     'AccuraciesSLC': accuraciesSLC,
            #     'AccuraciesSMARTSLC': accuraciesSMARTSLC,
            #     'x_data': flip_chances
            # })
            # savename = f"Results/{args.model}.csv"
            # result_data.to_csv(savename, index=False)

            # plt.figure()
            # plt.plot(flip_chances, accuraciesMLC, 'r-', label="MLC Only")
            # plt.plot(flip_chances, accuraciesSLC, 'g-', label="MLC + SLC")
            # plt.plot(flip_chances, accuraciesSMARTSLC, 'm-', label="SMART SLC")
            # plt.xscale('log')  # Set x-axis to be logarithmic
            # plt.xlabel("BER")
            # plt.ylabel("Accuracy")
            # plt.title("Accuracy vs Bit Error Rate")
            # plt.grid(which='both')  # This will enable grid lines for both major and minor ticks
            # plt.legend()
            # # if args.save is True:
            # savename = f"Results/{args.model}"
            # plt.savefig(savename)
            # else:




def linear_quantize(input, sf, bits):
    assert bits >= 1, bits
    if bits == 1:
        return torch.sign(input) - 1
    delta = math.pow(2.0, -sf)
    bound = math.pow(2.0, bits-1)
    min_val = - bound
    max_val = bound - 1
    rounded = torch.floor(input / delta + 0.5)
    clipped_value = torch.clamp(rounded, min_val, max_val)

    # clipped_value = torch.clamp(rounded, min_val, max_val) * delta
    return clipped_value, delta

def compute_integral_part(input, overflow_rate):
    abs_value = input.abs().view(-1)
    sorted_value = abs_value.sort(dim=0, descending=True)[0]
    split_idx = int(overflow_rate * len(sorted_value))
    v = sorted_value[split_idx]
    # if isinstance(v, Variable):
    #     v = v.data.cpu().numpy()[0]
    sf = math.ceil(math.log2(v+1e-12))
    return sf


# Training
def train(epoch):
    print("\nEpoch: %d" % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(
            batch_idx,
            len(trainloader),
            "Loss: %.3f | Acc: %.3f%% (%d/%d)"
            % (train_loss / (batch_idx + 1), 100.0 * correct / total, correct, total),
        )


def test(net, criterion, optimizer, testloader, device):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            progress_bar(
                batch_idx,
                len(testloader),
                "Loss: %.3f | Acc: %.3f%% (%d/%d)"
                % (
                    test_loss / (batch_idx + 1),
                    100.0 * correct / total,
                    correct,
                    total,
                ),
            )
    acc = 100.0 * correct / total
    return acc

if __name__ == "__main__":
    main()
