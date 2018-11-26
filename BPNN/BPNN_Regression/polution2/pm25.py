#-*- coding：utf-8 -*-
# &Author  AnFany

# 适用于多维输出
from HB_Data_Reg import model_data as R_data
from HB_Data_Reg import L as LL
from HB_Data_Reg import n as N
import numpy as np
import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import matplotlib.pyplot as plt

'''第一部分：数据准备'''


'''第二部分： 基于TensorFlow构建训练函数'''
# 创建激活函数
def activate(input_layer, weights, biases, actfunc):
    layer = tf.add(tf.matmul(input_layer, weights), biases)
    if actfunc == 'relu':
        return tf.nn.relu(layer)
    elif actfunc == 'tanh':
        return tf.nn.tanh(layer)
    elif actfunc == 'sigmoid':
        return tf.nn.sigmoid(layer)
# 权重初始化的方式和利用激活函数的关系很大
# sigmoid: xavir  tanh: xavir    relu: he

#  构建训练函数
def Ten_train(xdata, ydata,prexdata,key,hiddenlayers=3, hiddennodes=100, \
              learn_rate=0.05, itertimes=5000, batch_size=200, activate_func='sigmoid', break_error=0.0015):
    # 开始搭建神经网络
    Input_Dimen = len(xdata[0])
    Unit_Layers = [Input_Dimen] + [hiddennodes] * hiddenlayers + [len(ydata[0])]  # 输入的维数，隐层的神经数，输出的维数1

    # 创建占位符
    x_data = tf.placeholder(shape=[None, Input_Dimen], dtype=tf.float32)
    y_target = tf.placeholder(shape=[None, len(ydata[0])], dtype=tf.float32)

    # 实现动态命名变量
    VAR_NAME = locals()

    for jj in range(hiddenlayers + 1):
        VAR_NAME['weight%s' % jj] = tf.Variable(np.random.rand(Unit_Layers[jj], Unit_Layers[jj + 1]), dtype=tf.float32,\
                                                name='weight%s' % jj) / np.sqrt(Unit_Layers[jj])  # sigmoid  tanh
        # VAR_NAME['weight%s'%jj] = tf.Variable(np.random.rand(Unit_Layers[jj], Unit_Layers[jj + 1]), dtype=tf.float32,name='weight%s' % jj) \/ np.sqrt(Unit_Layers[jj] / 2)  # relu
        VAR_NAME['bias%s' % jj] = tf.Variable(tf.random_normal([Unit_Layers[jj + 1]], stddev=10, name='bias%s' % jj),
                                              dtype=tf.float32)
        if jj == 0:
            VAR_NAME['ooutda%s' % jj] = activate(x_data, eval('weight%s' % jj), eval('bias%s' % jj), actfunc=activate_func)
        else:
            VAR_NAME['ooutda%s' % jj] = activate(eval('ooutda%s' % (jj - 1)), eval('weight%s' % jj), \
                                                 eval('bias%s' % jj), actfunc=activate_func)

    # 均方误差
    loss = tf.reduce_mean(tf.reduce_sum(tf.square(y_target - eval('ooutda%s' % (hiddenlayers))), reduction_indices=[1]))

    # 优化的方法
    my_opt = tf.train.AdamOptimizer(learn_rate)
    train_step = my_opt.minimize(loss)

    # 初始化
    init = tf.global_variables_initializer()

    loss_vec = []  # 训练误差

    with tf.Session() as sess:
        saver = tf.train.Saver()
        sess.run(init)
        for i in range(itertimes):
            rand_index = np.random.choice(len(xdata), size=batch_size, replace=False)
            rand_x = xdata[rand_index]
            rand_y = ydata[rand_index]

            sess.run(train_step, feed_dict={x_data: rand_x, y_target: rand_y})

            temp_loss = sess.run(loss, feed_dict={x_data: xdata, y_target: ydata})

            loss_vec.append(temp_loss)

            # 根据输出的误差，判断训练的情况
            if (i + 1) % 25 == 0:
                print(key+' '+'Generation: ' + str(i + 1) + '. 归一误差：Loss = ' + str(temp_loss))

            # 提前退出的判断
            if temp_loss < break_error:  # 根据经验获得此数值, 因为采用的是随机下降，因此误差在前期可能出现浮动
                break

        # 计算预测数据的输出
        pre_in_data0 = np.array(prexdata, dtype=np.float32)
        for ipre in range(hiddenlayers + 1):
            VAR_NAME['pre_in_data%s' % (ipre + 1)] = activate(eval('pre_in_data%s' % ipre), eval('weight%s' % ipre).eval(),\
                                                                 eval('bias%s' % ipre).eval(), actfunc=activate_func)

        # 计算训练数据的输出
        train_in_data0 = np.array(xdata, dtype=np.float32)
        for ipre in range(hiddenlayers + 1):
            VAR_NAME['train_in_data%s' % (ipre + 1)] = activate(eval('train_in_data%s' % ipre), eval('weight%s' % ipre).eval(),\
                                                                 eval('bias%s' % ipre).eval(), actfunc=activate_func)
        # path = 'model2/'+key+'checkpoint/model.ckpt'
        # saver.save(sess,path)
        return eval('train_in_data%s'%(hiddenlayers+1)).eval(), eval('pre_in_data%s'%(hiddenlayers+1)).eval(), loss_vec

'''第三部分： 结果展示函数'''
#import matplotlib.pyplot as plt
from pylab import mpl  # 作图显示中文
mpl.rcParams['font.sans-serif'] = ['FangSong'] # 设置中文字体新宋体
#  绘制图像
def figure(real, net, le='训练', real_line='ko-', net_line='r.-', width=3):
    length = len(real[0])
    # 绘制每个维度的对比图
    for iwe in range(length):
        plt.subplot(length, 1, iwe+1)
        plt.plot(list(range(len(real.T[iwe]))), real.T[iwe], real_line, linewidth=width)
        plt.plot(list(range(len(net.T[iwe]))), net.T[iwe], net_line, linewidth=width - 1)
        plt.legend(['%s真实值'%le, '网络输出值'])
        if length == 1:
            plt.title('%s结果对比'%le)
        else:
            if iwe == 0:
                plt.title('%s结果: %s维度对比'%(le, iwe))
            else:
                plt.title('%s维度对比'%iwe)
    plt.show()

# 绘制成本函数曲线图
def costfig(errlist, le='成本函数曲线图'):
    plt.plot(list(range(len(errlist))), errlist, linewidth=3)
    plt.title(le)
    plt.xlabel('迭代次数')
    plt.ylabel('成本函数值')
    plt.show()

#  因为训练数据较多，为了不影响展示效果，按序随机选取一定数量的数据，便于展示
def select(datax, datay, count=200):
    sign = list(range(len(datax)))
    selectr_sign = np.random.choice(sign, count, replace=False)
    dx=datax[selectr_sign]
    dy=datay[selectr_sign]
    return dx,dy

# 将输出的数据转换尺寸，变为原始数据的尺寸
# def trans(ydata, minumber=R_data[4][0], maxumber=R_data[4][1]):
#     return ydata * (maxumber - minumber) + minumber

#处理训练和测试的输入数据
def InputHandler(dat,LL):
    rows = dat.shape[0]
    cols = dat.shape[1]
    data = []
    for index in range(rows-LL+1):
        dd = []
        for k in range(index,index+LL):
            dd.append(dat[k])
        dt = [0]*36
        for p in range(len(dd)):
            nd = dd[p]
            for q in range(cols):
                dt[q+p*cols] = nd[q]
        data.append(dt)
    return data
if __name__ == '__main__':
    errordic = {}
    data = R_data.get('PM2.5')
    train_x_data = data[0]  # 训练输入
    train_y_data = data[1]  # 训练输出
    predict_x_data = data[2]  # 测试输入
    predict_y_data = data[3]  # 测试输出
    for nodeNum in range(5,100):
        #  训练
        tfrelu = Ten_train(train_x_data, train_y_data, predict_x_data,'PM2.5',hiddenlayers=3, hiddennodes=nodeNum)
        minumber = data[4][1]
        maxumber = data[4][0]

        #训练和测试的真实输出
        train_y_data_tran = train_y_data*(maxumber - minumber) + minumber
        predict_y_data_tran = predict_y_data*(maxumber - minumber) + minumber

        #训练和测试的预测输出
        train_output = tfrelu[0] * (maxumber - minumber) + minumber
        predict_output = tfrelu[1] * (maxumber - minumber) + minumber
        error = [0] * len(predict_output)
        for index in range(len(predict_output)):
            error[index] = abs(predict_y_data_tran[index] - predict_output[index]) / predict_y_data_tran[index]
        count = 0
        for k in range(len(error)):
            if error[k] <= 0.3:
                count = count + 1
        auc = count / len(predict_output)
        errordic.setdefault(nodeNum, auc)
        if auc >= 0.75:
            break
    print(errordic)
        # 数据多影响展示，随机挑选100条数据
        # random_train_x_data = select(train_output, train_y_data_tran, 200)
        # random_predict_x_data = select(predict_output, predict_y_data_tran, 100)
        # figure(random_train_x_data[1], random_train_x_data[0], le='训练')
        # figure(random_predict_x_data[1], random_predict_x_data[0], le='预测')
        # costfig(tfrelu[2])



