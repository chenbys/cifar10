import argparse

parser = argparse.ArgumentParser(description='Cifar-10 Training')
parser.add_argument('--name', default='adaboost', type=str)
parser.add_argument('--pca_dim', default=32 * 10, type=int)
parser.add_argument('--n_estimators', default=100, type=int)
parser.add_argument('--max_depth', default=4, type=int)
parser.add_argument('--learning_rate', default=1.0, type=float)

args = parser.parse_args()

from models import ensemble
import os
import torchvision
import numpy as np
import cv2


def main():
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True)
    # convert to gray
    gray_train = [cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) for img in trainset.train_data]
    gray_test = [cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) for img in testset.test_data]

    adaboost = ensemble.Adaboost(np.array(gray_train) / 255., np.array(trainset.train_labels),
                                 np.array(gray_test) / 255., np.array(testset.test_labels), args)
    train_acc = adaboost.ctrain()
    test_acc = adaboost.ctest()
    print()
    print(train_acc)
    print(test_acc)
    return


if __name__ == '__main__':
    main()
