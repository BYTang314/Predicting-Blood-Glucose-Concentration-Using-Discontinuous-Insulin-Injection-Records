import tensorflow as tf
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# change the GPU id
# os.environ["CUDA_VISIBLE_DEVICES"] = "3" 

import numpy as np
import pandas
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor

from tensorflow.keras.callbacks import TensorBoard


from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from tensorflow.keras.layers import Conv1D,Flatten,MaxPooling1D
from allmodel import FCN1, FCN2


train_data = np.load("../../Data/traindata.npy")
test_data = np.load("../../Data/testdata.npy")


X = train_data[:,0:31]
Y = train_data[:,31]
test_X =  test_data[:,0:31]
test_Y = test_data[:,31]



print(X[0],Y[0],X.shape)

#####################################################################################
# change the model to train, FCN1, FCN2
model = FCN1()
model.summary()
# change the log path
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir="../../Log/FCN1")
# change the hyperparametrics(epochs, batch_size)
history = model.fit(X, Y, epochs=7000, batch_size=1000,verbose=True, callbacks=[tensorboard_callback])
# change the model save path
model.save("../../Model/FCN1")
######################################################################################


print("[INFO] predicting glc_after...")
pred_test_y = model.predict(test_X)
print(pred_test_y[0:10])
print(test_Y[0:10])

print("[INFO] predicting glc_after in trainset...")
pred_train_y = model.predict(X)
print(pred_train_y[10:20])
print(Y[10:20])


from sklearn.metrics import r2_score,mean_squared_error,mean_absolute_error
#r2
pred_acc = r2_score(test_Y, pred_test_y)
print('pred_acc_r2',pred_acc)
train_acc = r2_score(Y, pred_train_y)
print('train_acc_r2',train_acc)
#,mae
pred_acc = mean_absolute_error(test_Y, pred_test_y)
print('pred_acc_mae',pred_acc)
train_acc = mean_absolute_error(Y, pred_train_y)
print('train_acc_mae',train_acc)
#,mse
pred_acc = mean_squared_error(test_Y, pred_test_y)
print('pred_acc_mse',pred_acc)
train_acc = mean_squared_error(Y, pred_train_y)
print('train_acc_mse',train_acc)
#rmse
print('pred_acc_rmse',pred_acc**0.5)
print('train_acc_rmse',train_acc**0.5)
