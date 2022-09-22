import numpy as np
from .tree_base import WeightTree2
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support as f1


class SR2:
    def __init__(self):
        self.machines = []
        self.fy = None
        self.weights = None
        self.fx = None
        self.cms = []

    def __train_once__(self, train_x, train_y, max_depth=3):
        n = train_x.shape[0]
        if self.weights is None:
            self.fy = np.zeros(n)
            self.fx = np.zeros(n)
            self.weights = np.ones(n) / n
        wrt = WeightTree2(max_depth=max_depth)
        wrt.train(train_x, train_y, self.weights)

        f_m = 2 * wrt.predict_probability(train_x) - 1
        efy = np.exp(self.fy)
        p = 1 - 1 / (1 + efy)
        print(p)
        w1 = (p * (1 - p)).sum()
        w2 = (1 - p).sum()
        if w2 < 1e-3:
            cm = 1
            self.weights = np.ones(n) / n
            self.cms.append(cm)
            self.machines.append(wrt)
        else:
            cm = w1 / w2
            self.cms.append(cm)
            self.machines.append(wrt)
            self.fx = self.fx + cm * f_m
            self.fy = self.fx * (2 * np.array(train_y) - 1)
            efy = np.exp(self.fy)
            self.weights = 1 / (1 + efy)
            self.weights = self.weights / self.weights.sum()

    def train(self, train_x, train_y, max_depth=3, nround=5):
        for i in range(nround):
            print(i)
            self.__train_once__(train_x, train_y, max_depth=max_depth)

    def train_by_old(self, train_x, train_y, max_depth=3, nround=5, old_one=None):
        if old_one is not None:
            self.machines = old_one.machines
            self.fy = old_one.fy
            self.weights = old_one.weights
            self.fx = old_one.fx
        for _ in range(nround):
            print(_)
            self.__train_once__(train_x, train_y, max_depth=max_depth)
        return self

    def fun(self, test_x):
        f = np.zeros(test_x.shape[0])
        for i, machine in enumerate(self.machines):
            pre = 2 * machine.predict_probability(test_x) - 1
            f = f + self.cms[i] * pre
        return f

    def predict(self, test_x):
        f = self.fun(test_x)
        return np.round(f > 0)

    def accuracy(self, test_x, test_y):
        pre = self.predict(test_x)
        y = np.array(test_y).reshape(pre.shape)
        res = np.round(pre == y)
        acc = res.mean()
        return acc, res

    # F1评分
    def f1_rating(self, test_x, test_y):
        prediction = self.predict(test_x)
        y = np.array(test_y)
        precision = np.mean(y[prediction == 1])
        recalling = np.mean(prediction[y == 1])
        f1_rate = 2 * precision * recalling / (precision + recalling + 0.00001)
        print("F1评分", f1_rate)
        return f1_rate


class SRM:
    def __init__(self):
        self.workers = []

    def train(self, train_x, train_y, max_depth=3, nround=5):
        J = len(set(train_y))
        n = train_x.shape[0]
        # set y_matrix for encoding of y by one-hot
        y_matrix = np.zeros((n, J))
        for i, y_i in enumerate(train_y):
            y_matrix[i, int(np.round(y_i))] = 1
        # for all data sets train the machine DiscreteAdaBoostMH
        for j in range(J):
            # change the data
            mid_train_y = y_matrix[:, j]
            worker = SR2()
            worker.train(train_x, mid_train_y, max_depth=max_depth, nround=nround)
            # save the workers
            self.workers.append(worker)
        return self

    def train_by_old(self, train_x, train_y, max_depth=3, nround=5, old_one=None):
        if old_one is None or len(old_one.workers) == 0:
            return self.train(train_x, train_y, max_depth=max_depth, nround=nround)
        self.workers = old_one.workers
        J = len(set(train_y))
        n = train_x.shape[0]
        # set y_matrix for encoding of y by one-hot
        y_matrix = np.zeros((n, J))
        for i, y_i in enumerate(train_y):
            y_matrix[i, int(np.round(y_i))] = 1
        # for all data sets train the machine DiscreteAdaBoostMH
        for j in range(J):
            # change the data
            train_y = y_matrix[:, j]
            self.workers[j] = self.workers[j].train_by_old(train_x, train_y, max_depth=max_depth,
                                                           nround=nround, old_one=self.workers[j])
        return self

    # get fun_j, j = 1,...,J
    def get_fun(self, test_x):
        f_x = {}
        for j, worker in enumerate(self.workers):
            c_j = 'F_' + str(j)
            f_x[c_j] = worker.fun(test_x)
        return pd.DataFrame(f_x)

    # predict
    def predict(self, test_x):
        f_x = self.get_fun(test_x)
        return np.array(f_x).argmax(axis=1)

    # accuracy
    def accuracy(self, test_x, test_y):
        pre = self.predict(test_x)
        y = np.array(test_y).reshape(pre.shape)
        res = abs(y - pre) < 0.1
        acc = res.mean()
        return acc, res

    def f1_rating(self, test_x, test_y):
        pre = self.predict(test_x)
        return f1(test_y, pre)[2].mean()