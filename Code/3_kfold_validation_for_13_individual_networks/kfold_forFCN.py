import tensorflow as tf
import os
import sys
print(sys.argv)
os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[1]
import numpy as np
import pandas 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout,Flatten,Conv1D,MaxPool1D
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor

from tensorflow.keras.callbacks import TensorBoard

from sklearn.metrics import r2_score,mean_squared_error,mean_absolute_error

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
np.set_printoptions(suppress=True) 

all_data = np.load("../../Data/alldata.npy") 

all_X = all_data[:,0:31]
all_Y = all_data[:,31]
kf = KFold(n_splits=5)
print(kf.get_n_splits(all_X))

# FCN1
def FCN1(input_num = 31):
    model = Sequential()
    model.add(Dense(2048, input_dim=input_num,kernel_initializer='normal', activation='relu',name="dense0"))
    model.add(Dense(1536,kernel_initializer='normal', activation='relu',name="dense1"))
    model.add(Dense(1024,kernel_initializer='normal', activation='relu',name="dense2"))
    model.add(Dense(768,kernel_initializer='normal', activation='relu',name="dense3"))
    model.add(Dense(512,kernel_initializer='normal', activation='relu',name="dense4"))
    model.add(Dense(384,kernel_initializer='normal', activation='relu',name="dense5"))
    model.add(Dense(256,kernel_initializer='normal', activation='relu',name="dense6"))
    model.add(Dense(128, kernel_initializer='normal',activation='relu',name="dense7"))
    model.add(Dense(64,kernel_initializer='normal',activation='relu',name="dense8"))
    model.add(Dense(32,kernel_initializer='normal', activation='relu',name="dense9"))
    model.add(Dense(16,kernel_initializer='normal', activation='relu',name="dense10"))
    model.add(Dense(1,kernel_initializer='normal',name="dense11"))
    model.compile(optimizer="adam",loss="mse")
    model.summary()
    return model


# FCN2
def FCN2(input_num = 31):
    model = Sequential()
    model.add(Dense(1024,input_dim=input_num, kernel_initializer='normal', activation='relu',name="dense1"))
    model.add(Dense(512,kernel_initializer='normal', activation='relu',name="dense2"))
    model.add(Dense(400,kernel_initializer='normal', activation='relu',name="dense3"))
    model.add(Dense(256,kernel_initializer='normal', activation='relu',name="dense4"))
    model.add(Dense(128, kernel_initializer='normal',activation='relu',name="dense5"))
    model.add(Dense(72, kernel_initializer='normal',activation='relu',name="dense6"))
    model.add(Dense(64,kernel_initializer='normal',activation='relu',name="dense7"))
    model.add(Dense(32,kernel_initializer='normal', activation='relu',name="dense8"))
    model.add(Dense(16,kernel_initializer='normal', activation='relu',name="dense9"))
    model.add(Dense(1,kernel_initializer='normal',name="dense10"))
    model.compile(optimizer="adam",loss="mse")
    print(model.summary())
    return model

pred_r2_list = []
train_r2_list = []
pred_mae_list = []
train_mae_list = []
pred_mse_list = []
train_mse_list = []
pred_rmse_list = []
train_rmse_list = []
A = []
B = []
C = []
D = []
E = []

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



for k,(train,test) in enumerate(kf.split(all_X,all_Y)):
    print (k,(train,test))
    X = all_X[train]
    Y = all_Y[train]
    test_X = all_X[test]
    test_Y = all_Y[test]
    
    ##############################################
    # change the model for k=fold validation
    # FCN1, FCN2
    model = FCN1()
    ##############################################
    history = model.fit(X, Y, epochs=int(sys.argv[2]), batch_size=1000,verbose=False)
    
    print("[INFO] predicting glc_after...")
    pred_test_y = model.predict(test_X)
    print(pred_test_y[0:10])
    print(test_Y[0:10])

    print("[INFO] predicting glc_after in trainset...")
    pred_train_y = model.predict(X)
    print(pred_train_y[10:20])
    print(Y[10:20])

    print("fold :" ,k)
    #r2
    pred_acc = r2_score(test_Y, pred_test_y)
    print('pred_acc_r2',pred_acc)
    train_acc = r2_score(Y, pred_train_y)
    print('train_acc_r2',train_acc)
    train_r2_list.append(train_acc)
    pred_r2_list.append(pred_acc)
    #,mae
    pred_acc = mean_absolute_error(test_Y, pred_test_y)
    print('pred_acc_mae',pred_acc)
    train_acc = mean_absolute_error(Y, pred_train_y)
    print('train_acc_mae',train_acc)
    train_mae_list.append(train_acc)
    pred_mae_list.append(pred_acc)
    #,mse
    pred_acc = mean_squared_error(test_Y, pred_test_y)
    print('pred_acc_mse',pred_acc)
    train_acc = mean_squared_error(Y, pred_train_y)
    print('train_acc_mse',train_acc)
    train_mse_list.append(train_acc)
    pred_mse_list.append(pred_acc)
    #rmse
    print('pred_acc_rmse',pred_acc**0.5)
    print('train_acc_rmse',train_acc**0.5)
    train_rmse_list.append(train_acc**0.5)
    pred_rmse_list.append(pred_acc**0.5)
    # grid
    a,b,c,d,e = get_CEG_percentage(test_Y,pred_test_y)
    A.append(a)
    B.append(b)
    C.append(c)
    D.append(d)
    E.append(e)
print() 
print("mean acc:")
print('pred_acc_r2',pred_r2_list,": ",np.mean(pred_r2_list))
print('train_acc_r2',train_r2_list,": ",np.mean(train_r2_list))
print('pred_acc_mae',pred_mae_list,": ",np.mean(pred_mae_list))
print('train_acc_mae',train_mae_list,": ",np.mean(train_mae_list))
print('pred_acc_mse',pred_mse_list,": ",np.mean(pred_mse_list))
print('train_acc_mse',train_mse_list,": ",np.mean(train_mse_list))
print('pred_acc_rmse',pred_rmse_list,": ",np.mean(pred_rmse_list))
print('train_acc_rmse',train_rmse_list,": ",np.mean(train_rmse_list))
print("grid_a",A,np.mean(A))
print("grid_b",B,np.mean(B))
print("grid_c",C,np.mean(C))
print("grid_d",D,np.mean(D))
print("grid_e",E,np.mean(E))
