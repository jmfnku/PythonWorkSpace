# import tensorflow as tf
# import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
#
# meta_path = 'model2/COcheckpoint/model.ckpt.meta'
# model_path = 'model2/COcheckpoint/model.ckpt'
# saver = tf.train.import_meta_graph(meta_path)
#
# with tf.Session() as sess:
#     saver.restore(sess,model_path)
#     graph = tf.get_default_graph()
from sklearn.externals import joblib
from HB_BPNN import BPNN
from HB_Data_Reg import model_data as R_data
D=3
n=9
predict={'SO2':54,'CO':1.6,'NO2':71,'O3-1H':69,'O3-8H':71,'PM10':235,'PM2.5':117,'NO':3,'NOx':74}
w_forcast = {1:[33,13.7,1.6,1004],2:[41,13.6,1.04,1003.76],3:[43,13.8,2.32,1003.17]}
pName = ['SO2','CO','NO2','O3-1H','O3-8H','PM2.5','PM10','NO','NOx']
def predict_Polution(predict,pName):
    predictRes = {}
    for index in range(D):
        p_temp = {}
        if index == 0:
            for k in range(len(pName)): #九次循环得到一次预测结果，串行
                key = pName[k]
                temp = dataHandler(predict,key,index) #得到特征集
                path = 'model/'+key+'.model'
                p = joblib.load(path)
                res = p.predict(temp) #此key对应的预测结果
                p_temp.setdefault(key,res[0])
            predictRes.setdefault(index,p_temp)

        else:
            for k in range(len(pName)):
                key = pName[k]
                temp = dataHandler(predictRes[index-1], key,index)  # 得到特征集
                p = joblib.load('model/' + key + '.model')
                res = p.predict(temp)  # 此key对应的预测结果
                p_temp.setdefault(key,res[0])
            predictRes.setdefault(index,p_temp)
    return  predictRes

def dataHandler(predict,key,d):
    temp = []
    for k in predict:
        if k != key:
            temp.append(predict.get(k))
    w = w_forcast.get(d+1)
    for i in range(len(w)):
        temp.append(w[i])
    return temp
def revers(res):
    for i in range(D):
        for key in res[i]:
            min = R_data.get(key)[4][1]
            max = R_data.get(key)[4][0]
            x = res[i].get(key)
            t = x*(max - min)+min
            res[i][key] = t
if __name__ == "__main__":
    res = predict_Polution(predict,pName)
    revers(res)
    print(res)