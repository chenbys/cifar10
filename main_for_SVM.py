from __future__ import print_function

from models import svms
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

    svm = svms.SVM(np.array(gray_train) / 255., np.array(trainset.train_labels),
                   np.array(gray_test) / 255., np.array(testset.test_labels), pca_dim=32 * 10,
                   name='SVM-gray-255')
    train_acc = svm.ctrain()
    test_acc = svm.ctest()
    print()
    print(train_acc)
    print(test_acc)
    return


if __name__ == '__main__':
    main()
