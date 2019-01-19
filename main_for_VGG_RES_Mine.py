from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import argparse

from models import resnet
from models import vgg
from models import mine

from tqdm import tqdm
import logging
import datetime


def main():
    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
    parser.add_argument('--model_name', default='vgg16', type=str,
                        help='the string startswith means which model to use,  it is also log_name and save_name')
    parser.add_argument('--lr', default=1e-1, type=float)
    parser.add_argument('--wd', default=1e-4, type=float)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--mm', default=0.9, type=float, help='adam or sgd')
    parser.add_argument('--optim', default='sgd', type=str, help='adam or sgd')
    parser.add_argument('--resume', default=None, type=str, help='resume path to checkpoint, None for restart')
    # parser.add_argument('--resume', default='res18_train0.93_test0.89_eph80', type=str,
    #                     help='resume path to checkpoint, None for restart')
    parser.add_argument('--num_epoch', default=200, type=int)
    args = parser.parse_args()

    #############################################
    logger = logging.getLogger(__name__)
    logger.setLevel(level=logging.INFO)
    now = datetime.datetime.now().strftime('%m-%d@%H-%M')
    handler = logging.FileHandler(f'logs/{args.model_name}_{now}.log')
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s:    |%(message)s')
    handler.setFormatter(formatter)
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    logger.addHandler(handler)
    logger.addHandler(console)

    def log(msg='QAQ'):
        logger.info(str(msg))

    ###############################################
    log(args)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    log(device)

    # Data
    print('==> Preparing data..')
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=4)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=4)

    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    # Model
    log('==> Building model..')
    # net = vgg.VGG11('VGG11')
    if args.model_name.startswith('res18'):
        net = resnet.ResNet18()
    elif args.model_name.startswith('vgg16'):
        net = vgg.VGG('VGG16')
    elif args.model_name.startswith('mine_dropout'):
        net = mine.Mine_dropout()
    elif args.model_name.startswith('mine_1122_dropout'):
        net = mine.Mine_1122_dropout()
    elif args.model_name.startswith('mine_1111_dropout'):
        net = mine.Mine_1111_dropout()
    else:
        net = mine.Mine()

    net = net.to(device)
    if device == 'cuda':
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True

    if args.resume:
        # Load checkpoint.
        log('==> Resuming from checkpoint..')
        assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
        checkpoint = torch.load(f'./checkpoint/{args.resume}.ckpt')
        net.load_state_dict(checkpoint['net'])
        start_epoch = checkpoint['cur_epoch'] + 1
    else:
        log('restart')
        start_epoch = 0

    criterion = nn.CrossEntropyLoss()
    if args.optim == 'sgd':
        optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=args.mm, weight_decay=args.wd)
    else:
        optimizer = optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.wd)

    def train():
        net.train()
        train_loss = 0
        hit, total_size = 0, 0
        for batch_idx, (inputs, targets) in tqdm(enumerate(trainloader)):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            _, predicted = outputs.max(1)
            hit += predicted.eq(targets).sum().item()
            total_size += predicted.shape[0]
        return hit / total_size

    def test():
        net.eval()
        test_loss = 0
        hit, total_size = 0, 0
        with torch.no_grad():
            for batch_idx, (inputs, targets) in tqdm(enumerate(testloader)):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = net(inputs)
                _, predicted = outputs.max(1)
                hit += predicted.eq(targets).sum().item()
                total_size += predicted.shape[0]
        return hit / total_size

    def save(net, train_acc, test_acc, cur_epoch):
        log('Saving...')
        state = {
            'net'      : net.state_dict(),
            'train_acc': train_acc,
            'test_acc' : test_acc,
            'cur_epoch': cur_epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        save_path = f'./checkpoint/{args.model_name}_train{train_acc:.2f}_test{test_acc:.2f}_eph{cur_epoch}.ckpt'
        torch.save(state, save_path)
        log(save_path)

    for epoch in range(start_epoch, start_epoch + args.num_epoch):
        train_acc = train()
        test_acc = test()
        log(f'Epoch: {epoch}, train acc: {train_acc:.2f}, test acc: {test_acc:.2f}')
        if (epoch + 1) % 40 == 0:
            save(net, train_acc, test_acc, epoch)
        # log('^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^')

    return


if __name__ == '__main__':
    main()
