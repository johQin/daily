import numpy as np
import pandas as pd
import tensorflow as tf


df = pd.read_csv('./data/boston.csv')
df = df.values
df = np.array(df)
for i in range(12):
    df[:, i] = df[:, i]/(df[:, i].max()-df[:, i].min())
y_data = df[:, 12]
x_data = df[:, :12]

sess = tf.Session()
saver = tf.train.import_meta_graph('./model/model-1616402671.meta', clear_devices=True)
saver.restore(sess, './model/model-1616402671')
pred_byname = tf.get_default_graph().get_operation_by_name('Model/pred_opt').outputs[0]
pred_bycol = tf.get_collection('pred_col')[0]
n = np.random.randint(506)
print('n:', n)
x_test = x_data[n].reshape(1, 12)
resp_byname = sess.run(pred_byname,feed_dict={'X:0': x_test})
resp_bycol = sess.run(pred_bycol,feed_dict={'X:0': x_test})
target = y_data[n]
print("标签值：%f" % target )
print("pred_byname预测值：%f" % resp_byname)
print("pred_bycol预测值：%f" % resp_bycol)



