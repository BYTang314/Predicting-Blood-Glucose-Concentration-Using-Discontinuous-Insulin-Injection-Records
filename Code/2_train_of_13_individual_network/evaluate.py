import tensorflow as tf
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0" # 使用第二块GPU（从0开始）
import numpy as np
import pandas
import sys
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor
from tensorflow.keras.callbacks import TensorBoard
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from tensorflow.keras.layers import Conv1D,Flatten,MaxPooling1D

model_name = sys.argv[1]
train_data = np.load("../../Data/traindata.npy")
test_data = np.load("../../Data/testdata.npy")


X = train_data[:,0:31]
Y = train_data[:,31]
test_X =  test_data[:,0:31]
test_Y = test_data[:,31]
# for CNN
if "CNN" in model_name:
    X = np.expand_dims(X, axis=2)
    test_X = np.expand_dims(test_X, axis=2)
# for LSTM
if "LSTM" in model_name:
    X = np.expand_dims(X, axis=1)
    test_X = np.expand_dims(test_X, axis=1)



print(X[0],Y[0],X.shape)


from tensorflow import keras
# change model path
model = keras.models.load_model("../../Model/" + model_name)



print("[INFO] predicting glc_after...")
pred_test_y = model.predict(test_X)
print(pred_test_y[0:10])
print(test_Y[0:10])

print("[INFO] predicting glc_after in trainset...")
pred_train_y = model.predict(X)
print(pred_train_y[10:20])
print(Y[10:20])


def get_CEG_percentage(test_Y,pred_test_y):
    total = test_Y.shape[0]
    re = np.zeros(5)
    for i in range(total):
        if (pred_test_y[i]<=70 and test_Y[i]<=70) or (pred_test_y[i]<=1.2*test_Y[i] and pred_test_y[i] >= 0.8*test_Y[i]):
            re[0] = re[0] + 1       #Zone A
        elif (test_Y[i]>=180 and pred_test_y[i]<=70) or (test_Y[i]<=70 and  pred_test_y[i]>=180):
            re[4] = re[4] + 1      # Zone E
        elif ((test_Y[i] >= 70 and test_Y[i] <= 290) and (pred_test_y[i] >= test_Y[i] + 110) ) or ((test_Y[i] >= 130 and test_Y[i] <= 180)and (pred_test_y[i] <= (7/5)*test_Y[i] - 182)):
            re[2] = re[2] + 1       # Zone C
        elif (( test_Y[i] >= 240) and ((pred_test_y[i] >= 70) and (pred_test_y[i] <= 180))) or (test_Y[i] <= 175/3 and (pred_test_y[i] <= 180) and (pred_test_y[i] >= 70)) or ((test_Y[i] >= 175/3 and test_Y[i] <= 70) and (pred_test_y[i] >= (6/5)*test_Y[i])):
            re[3] = re[3] + 1      # Zone D
        else:
            re[1] = re[1] + 1      # Zone
    percentage = re/total*100;
    print(percentage)
    return percentage[0], percentage[1], percentage[2], percentage[3], percentage[4]


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
a,b,c,d,e = get_CEG_percentage(test_Y,pred_test_y)
print(a,b,c,d,e)
