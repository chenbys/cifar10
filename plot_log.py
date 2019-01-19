import matplotlib.pyplot as plt
import numpy as np


def plot_log(log_file='logs/mine_01-17@15-03.log'):
    train_acc, test_acc = [], []
    with open(log_file) as f:
        for line in f.readlines():
            main = line.strip().split('|')[1]
            if main.startswith('Epoch'):
                t = main.split('train acc: ')[1]
                train, test = t.split(', test acc: ')
                train_acc.append(float(train))
                test_acc.append(float(test))

    f = plt.figure()
    a1 = f.add_subplot(111)

    a1.plot(train_acc, linewidth=3, label='train')
    a1.plot(test_acc, linewidth=3, label='test')
    a1.legend(loc='bottom right', fontsize=10)

    plt.show()


# plot_log()


def plot_comapre_log(log_files=['logs/res18_1_01-10@15-29.log'], names=['res18']):
    train_accs, test_accs = [], []
    for log_file in log_files:
        train_acc, test_acc = [], []
        with open(log_file) as f:
            for line in f.readlines():
                main = line.strip().split('|')[1]
                if main.startswith('Epoch'):
                    t = main.split('train acc: ')[1]
                    train, test = t.split(', test acc: ')
                    train_acc.append(float(train))
                    test_acc.append(float(test))
        train_accs.append(train_acc)
        test_accs.append(test_acc)

    f = plt.figure()
    a1 = f.add_subplot(121)
    a2 = f.add_subplot(122)
    for train_acc, name in zip(train_accs, names):
        a1.plot(train_acc, linewidth=3, label=name)
        a1.legend(loc='bottom right', fontsize=10)

    for test_acc, name in zip(test_accs, names):
        a2.plot(test_acc, linewidth=3, label=name)
        a2.legend(loc='bottom right', fontsize=10)

    plt.show()


# logs = ['A:/workspace/Cifar-10/logs/res18_1_01-10@15-29.log',
#         'A:/workspace/Cifar-10/logs/res18_2_01-10@19-01.log',
#         'A:/workspace/Cifar-10/logs/res18_3_01-10@22-48.log',
#         'A:/workspace/Cifar-10/logs/res18_4_01-11@03-27.log',
#         'A:/workspace/Cifar-10/logs/res18_5_01-11@07-21.log',
#         'A:/workspace/Cifar-10/logs/res18_6_01-11@09-43.log']
# names = ['sgd, wd:0, lr:1e-1',
#          'sgd, wd:5e-4, lr:1e-1',
#          'sgd, wd:5e-4, lr:1e-3',
#          'sgd, wd:5e-4, lr:1e-4',
#          'adam, wd:0, lr:1e-3',
#          'adam, wd:5e-4, lr:1e-3']

# logs = ['A:/workspace/Cifar-10/logs/vgg16_1_01-11@13-43.log',
#         'A:/workspace/Cifar-10/logs/vgg16_2_01-11@15-36.log',
#         'A:/workspace/Cifar-10/logs/vgg16_3_01-11@17-34.log',
#         'A:/workspace/Cifar-10/logs/vgg16_4_01-11@20-15.log',
#         'A:/workspace/Cifar-10/logs/vgg16_5_01-11@23-07.log',
#         'A:/workspace/Cifar-10/logs/vgg16_6_01-12@01-55.log']
# names = ['sgd, wd:0, lr:1e-1',
#          'sgd, wd:5e-4, lr:1e-1',
#          'adam, wd:5e-4, lr:1e-3',
#          'adam, wd:5e-4, lr:1e-4',
#          'adam, wd:0, lr:1e-3',
#          'adam, wd:5e-4, lr:1e-3']
# plot_comapre_log(logs, names)
# logs = ['A:/workspace/Cifar-10/logs/mine_dropout_01-19@00-24.log',
#         'A:/workspace/Cifar-10/logs/mine_dropout_01-19@04-48.log',
#         'A:/workspace/Cifar-10/logs/mine_dropout_01-19@09-17.log',
#         'A:/workspace/Cifar-10/logs/mine_dropout_01-19@11-19.log',
#         'A:/workspace/Cifar-10/logs/mine_dropout_01-19@15-33.log']
# names = ['wd:0',
#          'wd:1e-2',
#          'wd:1e-3',
#          'wd:1e-4',
#          'wd:1e-5']
# plot_comapre_log(logs, names)
# logs = ['A:\workspace\Cifar-10\logs\mine_1111_dropout_01-19@00-25.log',
#         'A:\workspace\Cifar-10\logs\mine_1111_dropout_01-19@04-14.log',
#         'A:\workspace\Cifar-10\logs\mine_1111_dropout_01-19@08-03.log',
#         'A:\workspace\Cifar-10\logs\mine_1111_dropout_01-19@11-19.log',
#         'A:\workspace\Cifar-10\logs\mine_1111_dropout_01-19@15-13.log']
# names = ['wd:0',
#          'wd:1e-2',
#          'wd:1e-3',
#          'wd:1e-4',
#          'wd:1e-5']
# plot_comapre_log(logs, names)
logs = ['A:\workspace\Cifar-10\logs\mine_1122_dropout_01-19@00-25.log',
        'A:\workspace\Cifar-10\logs\mine_1122_dropout_01-19@07-12.log',
        'A:\workspace\Cifar-10\logs\mine_1122_dropout_01-19@11-17.log',
        'A:\workspace\Cifar-10\logs\mine_1122_dropout_01-19@11-18.log',
        'A:\workspace\Cifar-10\logs\mine_1122_dropout_01-19@15-33.log']
names = ['wd:0',
         'wd:1e-2',
         'wd:1e-3',
         'wd:1e-4',
         'wd:1e-5']
plot_comapre_log(logs, names)
