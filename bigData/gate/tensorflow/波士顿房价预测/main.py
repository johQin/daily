import tensorflow as tf
import numpy as np
import pandas as pd
import sys
import os
import time
from sklearn.utils import shuffle
from matplotlib import pyplot as plt
# 1. 预处理数据
df = pd.read_csv('./data/boston.csv')
# print(df)
# print(df.describe())
df = df.values
df = np.array(df)
for i in range(12):
    df[:, i] = df[:, i]/(df[:, i].max()-df[:, i].min())
y_data = df[:, 12]
x_data = df[:, :12]
# print(plt)
# plt.plot(y_data)
# plt.show()

# 2.建模
x = tf.placeholder(tf.float32, [None, 12], name="X")
y = tf.placeholder(tf.float32, [None, 1], name="Y")

with tf.name_scope("Model"):
    w = tf.Variable(tf.random_normal([12, 1], stddev=0.01), name='W')
    b = tf.Variable(1.0, name='b')
    def model(x, w, b):
        return tf.matmul(x, w)+b
    pred = tf.identity(model(x, w, b), name='pred_opt')
    tf.add_to_collection("pred_col", pred)


print(sys.argv[1])
is_train = sys.argv[1]

sess = tf.Session()
ckpt_dir = "./model"

if is_train == '0':
    print('model')
    saver = tf.train.Saver()
    ckpt = tf.train.get_checkpoint_state(ckpt_dir)
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)
        print("Restore model from " + ckpt.model_checkpoint_path)

        n = np.random.randint(506)
        print('n:', n)
        x_test = x_data[n].reshape(1, 12)
        predict = sess.run(pred, feed_dict={x: x_test})
        target = y_data[n]
        print("预测值：%f，标签值：%f" % (predict, target))
else:
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)
    # 声明完所有变量后，调用tf.train.Saver()
    # 用于训练完模型后，保存模型
    saver = tf.train.Saver()

    # 3.模型训练
    train_epochs = 50
    learning_rate = 0.01
    with tf.name_scope("LOSS_Function"):
        loss_function=tf.reduce_mean(tf.pow(y-pred,2))
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss_function)
    init = tf.global_variables_initializer()
    logdir = 'log'
    # tensorboard：
    # tb_step1：添加记录节点
    sum_loss_op = tf.summary.scalar("loss", loss_function)
    # tb_step2：汇总记录节点
    merged = tf.summary.merge_all()

    # 开始训练

    try:
        sess.run(init)
        # tb_step4：实例化书写器
        writer = tf.summary.FileWriter(logdir, sess.graph)
        loss_list = []
        for epoch in range(train_epochs):
            loss_sum = 0.0
            for xs, ys in zip(x_data, y_data):
                xs = xs.reshape(1, 12)
                ys = ys.reshape(1, 1)
                # tb_step3：运行汇总节点
                _, summary_str, loss = sess.run([optimizer, merged, loss_function], feed_dict={x: xs, y: ys})
                # tb_step5：书写器写入日志
                writer.add_summary(summary_str, epoch)
                loss_sum = loss_sum + loss
            shuffle(x_data, y_data)# 会改变原数组的顺序
            b0temp = b.eval(session=sess)
            w0temp = w.eval(session=sess)
            loss_average = loss_sum / len(y_data)
            print(loss_average)
            loss_list.append(loss_average)
            print("epoch=",epoch+1,"  loss=",loss_average, "  b=",b0temp, "  w=",w0temp)
        saver.save(sess, os.path.join(ckpt_dir, "model-{}".format(int(time.time()))))
        # tb_step5：关闭书写器
        writer.close()
        n = np.random.randint(506)
        print('n:', n)
        x_test = x_data[n].reshape(1, 12)
        predict = sess.run(pred, feed_dict={x: x_test})
        target = y_data[n]
        print("预测值：%f，标签值：%f" % (predict, target))
        plt.plot(loss_list)
        plt.show()

    finally:
        sess.close()


