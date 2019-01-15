from sklearn import svm
import logging
import datetime

logger = logging.getLogger(__name__)
logger.setLevel(level=logging.INFO)
now = datetime.datetime.now().strftime('%m-%d@%H-%M')
handler = logging.FileHandler(f'logs/SVM_{now}.log')
handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s:    |%(message)s')
handler.setFormatter(formatter)
console = logging.StreamHandler()
console.setLevel(logging.INFO)
logger.addHandler(handler)
logger.addHandler(console)


def log(msg='QAQ'):
    logger.info(str(msg))


class SVM(object):
    def __init__(self, train_data, train_label, test_data, test_label, name='SVM', pca_dim=64 * 20):
        self.model = svm.SVC(gamma='scale')
        self.name = f'{name}scale-raw'
        self.pca_preprocess(train_data, train_label, test_data, test_label, n_dim=pca_dim)

    def ctrain(self):
        import time
        print('begin')
        t1 = time.time()
        self.model.fit(self.train_data, self.train_label)
        t2 = time.time()
        log(f'train size:{len(self.train_label)}, cost {(t2-t1)/60:.2f}')
        train_acc = (self.model.predict(self.train_data) == self.train_label).sum() / len(self.train_label)
        log(f'train acc:{train_acc}')
        self.name = f'{self.name}-train{train_acc:.2f}@{(t2-t1)}s'
        return train_acc

    def pca_preprocess(self, train_data, train_label, test_data, test_label, n_dim):
        from preprocess import by_kernel_pca
        from preprocess import by_pca
        from preprocess import normlize
        from preprocess import row

        import numpy as np

        train_data = train_data.reshape((train_data.shape[0], -1))
        test_data = test_data.reshape((test_data.shape[0], -1))

        all_data = np.concatenate((train_data, test_data))
        pca_all_data, pca_ratio = row(all_data, n_dim)
        self.name = f'{self.name}-pcr{n_dim}@{pca_ratio:.2f}'
        # self.name = f'{self.name}-normlized'
        pca_train_data, pca_test_data = pca_all_data[0:train_data.shape[0]], pca_all_data[train_data.shape[0]:]
        idx = np.random.permutation(len(train_label))
        train_data, train_label = pca_train_data[idx], train_label[idx]

        self.train_data = train_data
        self.train_label = train_label
        self.test_data = pca_test_data
        self.test_label = test_label

    def ctest(self):
        res = self.model.predict(self.test_data)
        hit = (res == self.test_label).sum()
        acc = hit / len(self.test_label)
        log(f'test svm acc: {acc:.4f} @[{hit}/{len(self.test_label)}]')
        self.name = f'{self.name}-test{acc:.2f}'
        return acc
