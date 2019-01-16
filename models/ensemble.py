from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
import time
from sklearn import svm
import logging
import datetime

logger = logging.getLogger(__name__)
logger.setLevel(level=logging.INFO)
now = datetime.datetime.now().strftime('%m-%d_%H-%M')
handler = logging.FileHandler(f'logs/Adaboost_{now}.log')
handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s:    |%(message)s')
handler.setFormatter(formatter)
console = logging.StreamHandler()
console.setLevel(logging.INFO)
logger.addHandler(handler)
logger.addHandler(console)


def log(msg='QAQ'):
    logger.info(str(msg))


class Adaboost(object):
    def __init__(self, train_data, train_label, test_data, test_label, args):
        self.name = args.name
        log(args)
        self.pca_preprocess(train_data, train_label, test_data, test_label, n_dim=args.pca_dim)
        dt_stump = DecisionTreeClassifier(max_depth=args.max_depth, min_samples_leaf=1)
        dt_stump.fit(self.train_data, self.train_label)
        dt_stump_acc = dt_stump.score(self.train_data, self.train_label)
        log(f'stump acc: {dt_stump_acc}')

        self.model = AdaBoostClassifier(
            base_estimator=dt_stump,
            learning_rate=args.learning_rate,
            n_estimators=args.n_estimators,
            algorithm="SAMME.R")

    def ctrain(self):
        t1 = time.time()
        self.model.fit(self.train_data, self.train_label)
        log(f'time for train: {(time.time()-t1)/60:.1f}m')
        train_acc = (self.model.predict(self.train_data) == self.train_label).sum() / len(self.train_label)
        log(f'train acc:{train_acc}')
        return

    def pca_preprocess(self, train_data, train_label, test_data, test_label, n_dim):
        t1 = time.time()

        from preprocess import by_kernel_pca
        from preprocess import by_pca
        from preprocess import normlize
        from preprocess import row

        import numpy as np

        train_data = train_data.reshape((train_data.shape[0], -1))
        test_data = test_data.reshape((test_data.shape[0], -1))

        all_data = np.concatenate((train_data, test_data))
        pca_all_data, pca_ratio = by_pca(all_data, n_dim)
        pca_train_data, pca_test_data = pca_all_data[0:train_data.shape[0]], pca_all_data[train_data.shape[0]:]
        idx = np.random.permutation(len(train_label))
        train_data, train_label = pca_train_data[idx], train_label[idx]

        self.train_data = train_data
        self.train_label = train_label
        self.test_data = pca_test_data
        self.test_label = test_label
        log(f'finished preprocess: PCA: {n_dim}@{pca_ratio:.2f}, cost time: {(time.time()-t1)/60:.1f}m')

    def ctest(self):
        res = self.model.predict(self.test_data)
        hit = (res == self.test_label).sum()
        acc = hit / len(self.test_label)
        log(f'test adaboost acc: {acc:.4f} @[{hit}/{len(self.test_label)}]')
        return acc
