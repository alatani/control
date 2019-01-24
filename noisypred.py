# coding: utf-8

from dataclasses import dataclass
import sys
import numpy as np
import numpy.random

import sklearn
import sklearn.metrics
from sklearn.model_selection import cross_val_score

from typing import Callable


# 個人でも簡単に実験できます。
# 適当な入力変数100次元くらい用意して、そのうち10個だけがrelevantなXで y = f(x)+ε の関数 fを何らか生成後、
# f(x) と ε の分散比が1:1000くらいになるようにした人工ランダムデータを生成。
# あとは自分のアルゴが回帰した関数が真のf にどれくらい近いか計ってみればよい


@dataclass
class Dataset:
    samples: int
    sn_ratio: float
    true_f: Callable[[np.ndarray], np.ndarray]

    def __init__(self, true_f):
        pass


class TrueFunc:
    def __init__(self, relevant_dim, dim):
        absrate = 0.5
        self.dim = dim
        self.relevant_dim = relevant_dim
        self.absdim = int(relevant_dim * absrate)
        self.w = numpy.random.randint(2, size=relevant_dim) * 2 - 1

    # x: (sample, dim)
    # y: sample
    def __call__(self, x: np.ndarray):
        # y = X w
        y = np.dot(x[:, :self.relevant_dim], self.w[:])

        return y


def add_normalized_noise(ys: np.ndarray, sn_ratio):
    std = np.std(ys)
    ys[:] /= std
    ys[:] *= np.sqrt(sn_ratio)

    new_ys = ys.copy()
    new_ys[:] += np.random.normal(size=ys.shape)
    return ys, new_ys


def synthetic_data(sn_ratio=0.001, relevant_x_ratio=0.1):
    pass


# true_y: (sample, dim)
def evaluate(true_y: np.ndarray, y: np.ndarray):
    return sklearn.metrics.mean_squared_error(true_y, y)


def generate_dataset():
    dim = 100
    relevant_dim = 10
    sn_ratio = 0.001
    sample = 10000

    f = TrueFunc(relevant_dim, dim)
    X = numpy.random.normal(size=(sample, dim))
    true_y = f(X)
    true_y, y = add_normalized_noise(true_y, sn_ratio=sn_ratio)

    return X, true_y, y


if __name__ == "__main__":
    print(sys.version)

    X, true_y, y = generate_dataset()

    from sklearn.linear_model import LinearRegression
    from sklearn.svm import SVR
    from sklearn.model_selection import KFold

    kf = KFold(n_splits=10)
    scores = []
    count = 1
    for train, test in kf.split(X, y):
        print(count);
        count += 1
        # model = SVR(kernel="linear", C=0.01, epsilon=0.1)
        model = LinearRegression()
        model.fit(X[train, :], y[train])
        pred_y = model.predict(X[test, :])

        score = sklearn.metrics.mean_squared_error(y[test], pred_y)

        true_y_std = np.std(true_y)
        score = sklearn.metrics.mean_squared_error(true_y[test] / true_y_std, pred_y / true_y_std)
        # print(pred_y, y[test])
        # print(y[test])
        scores.append(score)

    # best_score = sklearn.metrics.accuracy_score(np.sign(true_y), np.sign(y))
    best_score = sklearn.metrics.mean_squared_error(true_y, numpy.zeros(y.shape))

    print(max(true_y))
    print(min(true_y))
    print(y.shape)
    print(true_y.shape)
    print("predition score:", np.mean(np.array(scores)))
    print("best score:", best_score)

    # print(f(np.ones(dim)*-2))
    # print(f(np.ones(dim)*-1))
    # print(f(np.ones(dim)*1))
    # print(f(np.ones(dim)*2))

    pass
