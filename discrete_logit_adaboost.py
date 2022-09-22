import numpy as np
from .tree_base import WeightTree2
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support as f1


class SD2:
    def __init__(self):
        self.machines = []
        self.cms = []
        self.fy = None
        self.weights = None

    @classmethod
    def bi_section(cls, low, high, f, threshold=1e-4):
        times = 0
        while times < 2000:
            mid = low + abs(high - low) / 2.0
            xia = f(low)
            shang = f(high)
            if xia * shang > 0:
                low -= 1
                high += 1
                continue

            zhong = f(mid)
            if abs(zhong) < threshold:
                return mid
            if abs(high - low) < threshold:
                return mid
            if zhong * shang < 0:
                low = mid
            if xia * zhong < 0:
                high = mid
            times += 1
            # print(times, low, xia, mid, zhong, high, shang)
        return mid

    def train_once(self, train_x, train_y, max_depth=3):
        if self.weights is None:
            self.weights = np.ones(train_x.shape[0]) / train_x.shape[0]
            self.fy = np.zeros(train_x.shape[0])
        machine = WeightTree2(max_depth=max_depth)
        machine.train(train_x, train_y, weights=self.weights)

        # cm
        acc, res = machine.accuracy(train_x, train_y)
        err = self.weights[res == 0].sum()
        if err < 1e-3:
            self.weights = np.ones(train_x.shape[0]) / train_x.shape[0]
            c_0 = 1
        elif err >= 0.5:
            self.weights = np.ones(train_x.shape[0]) / train_x.shape[0]
            return None
        else:
            res_s = 2 * res - 1

            def c_function(c):
                eyf = np.exp(self.fy + c * res_s)
                return (res_s / (1 + eyf)).sum()
            c_0 = self.bi_section(-1, 10, c_function)
        if c_0 > 0:
            self.cms.append(c_0)
            # fy
            self.fy = self.fy + c_0 * (2 * res - 1)
            self.weights = 1 / (1 + np.exp(self.fy))
            self.weights = self.weights / self.weights.sum()
            self.machines.append(machine)

    def train(self, train_x, train_y, max_depth=3, nround=5):
        for i in range(nround):
            print(i)
            self.train_once(train_x=train_x, train_y=train_y, max_depth=max_depth)
        return self

    def train_by_old(self, train_x, train_y, max_depth=3, nround=5, old_one=None):
        if old_one is not None or len(old_one.machines) > 0:
            self.machines = old_one.machines
            self.cms = old_one.cms
            self.fy = old_one.fy
            self.weights = old_one.weights
        self.train(train_x, train_y, max_depth=max_depth, nround=nround)
        return self

    def fun(self, test_x):
        f = np.zeros(test_x.shape[0])
        for i, machine in enumerate(self.machines):
            pre = machine.predict(test_x)
            f = f + self.cms[i] * (2 * pre - 1)
        return f

    def predict(self, test_x):
        f = self.fun(test_x)
        return (np.sign(f) + 1) / 2

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




class SDM:
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
            worker = SD2()
            worker.train(train_x, mid_train_y, max_depth=max_depth, nround=nround)
            # save the workers
            self.workers.append(worker)
        return self

    def train_by_old(self, train_x, train_y, max_depth=3, nround=5, old_one=None):
        if old_one is None or len(old_one.workers) < 1:
            return self.train(train_x, train_y, max_depth, nround=nround)
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
            mid_train_y = y_matrix[:, j]
            self.workers[j] = self.workers[j].train_by_old(train_x, mid_train_y,
                                                           max_depth=max_depth, nround=nround, old_one=self.workers[j])
        return self

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
        y = np.array(test_y)
        pre = self.predict(test_x)
        pre = pre.reshape(y.shape)
        res = abs(pre - y) < 0.1
        acc = res.mean()
        return acc, res

    def f1_rating(self, test_x, test_y):
        pre = self.predict(test_x)
        return f1(test_y, pre)[2].mean()
