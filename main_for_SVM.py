from __future__ import print_function

from models import svms
import os
import torchvision
import numpy as np


def main():
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True)
    svm = svms.SVM(trainset.train_data, np.array(trainset.train_labels),
                   testset.test_data, np.array(testset.test_labels))
    train_acc = svm.ctrain()
    test_acc = svm.ctest()
    print()
    print(train_acc)
    print(test_acc)
    return


if __name__ == '__main__':
    main()
