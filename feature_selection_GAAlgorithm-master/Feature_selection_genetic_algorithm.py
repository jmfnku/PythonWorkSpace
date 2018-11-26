# -*- encoding: utf-8 -*-

import random
import math
import numpy as np
import lightgbm as lgb
import pandas as pd
from Genetic_algorithm import GA
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score

class FeatureSelection(object):
    def __init__(self, aLifeCount=12):
        self.columns = ['SO2', 'CO', 'NO2', 'O3-1H', 'O3-8H', 'PM10', 'PM2.5', 'NO', 'NOx', '湿度', '温度', '风速', '大气压']
        basestation1 = "D://GannData//train.csv"
        basestation2 = "D://GannData//validate.csv"
        self.train_data = pd.read_csv(basestation1, low_memory=False, usecols=self.columns,encoding='ANSI')
        self.validate_data = pd.read_csv(basestation2, low_memory=False, usecols=self.columns,encoding='ANSI')
        self.lifeCount = aLifeCount
        self.ga = GA(aCrossRate=0.7,
                     aMutationRage=0.1,
                     aLifeCount=self.lifeCount,
                     aGeneLenght=len(self.columns) - 1,
                     aMatchFun=self.matchFun())
    def auc_score(self, order):
        print(order)
        features = self.columns[1:]
        features_name = []
        for index in range(len(order)):
            if order[index] == 1:
                features_name.append(features[index])

        labels = np.array(self.train_data['SO2'], dtype=np.int8)
        d_train = lgb.Dataset(self.train_data[features_name], label=labels)
        params = {
            'boosting': 'gbdt',
            'objective': 'binary',
            'metric': 'auc',
            'train_metric': False,
            'subsample': 0.8,
            'learning_rate': 0.05,
            'num_leaves': 96,
            'num_threads': 4,
            'max_depth': 5,
            'colsample_bytree': 0.8,
            'lambda_l2': 0.01,
            'verbose': -1,     # inhibit print info #
        }
        rounds = 500
        watchlist = [d_train]
        bst = lgb.train(params=params, train_set=d_train, num_boost_round=rounds, valid_sets=watchlist, verbose_eval=10)
        predict = bst.predict(self.validate_data[features_name])
        print(features_name)
        d1 = np.array(self.validate_data['SO2'])
        max = np.max(d1)
        min = np.min(d1)
        for i in range(len(d1)):
            d1[i] = (d1[i] - min)/(max-min)
            if d1[i]<0.5:
                d1[i] = 0
            else:
                d1[i] = 1
        d2 = predict
        score = roc_auc_score(d1,d2)
        print('validate score:', score)
        return score
    def matchFun(self):
        return lambda life: self.auc_score(life.gene)
    def run(self, n=0):
        distance_list = []
        generate = [index for index in range(1, n + 1)]
        while n > 0:
            self.ga.next()
            # distance = self.auc_score(self.ga.best.gene)
            distance = self.ga.score                      ####
            distance_list.append(distance)
            print(("第%d代 : 当前最好特征组合的线下验证结果为：%f") % (self.ga.generation, distance))
            n -= 1

        print('当前最好特征组合:')
        string = []
        flag = 0
        features = self.columns[1:]
        for index in self.ga.gene:
            if index == 1:
                string.append(features[flag])
            flag += 1
        print(string)
        print('线下最高为auc：', self.ga.score)

        '''画图函数'''
        plt.plot(generate, distance_list)
        plt.xlabel('generation')
        plt.ylabel('distance')
        plt.title('generation--auc-score')
        plt.show()


def main():
    fs = FeatureSelection(aLifeCount=20)
    rounds = 20    # 算法迭代次数 #
    fs.run(rounds)


if __name__ == '__main__':
    main()


