import tensorflow as tf
import os
import sys
print(sys.argv)
#os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[1]
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
from tensorflow.keras.layers import LSTM,TimeDistributed
from tensorflow.keras.layers import Conv1D,Flatten,MaxPooling1D


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

# CNN1
def CNN1(input_num = 31):
    model = Sequential()
    model.add(Conv1D(16, kernel_size=3, activation='relu',input_shape=(input_num,1),name="conv0"))
    model.add(Conv1D(64, 3, activation='relu',name="conv1"))
    model.add(Conv1D(128, 3, activation='relu',name="conv2"))
    model.add(Conv1D(256,3, activation='relu',name="conv3"))
    model.add(Conv1D(512,3, activation='relu',name="conv4"))
    model.add(Conv1D(256, 3, activation='relu',name="conv5"))
    model.add(Conv1D(128, 3, activation='relu',name="conv6"))
    model.add(Conv1D(64, 3, activation='relu',name="conv7"))
    model.add(Conv1D(16, 3, activation='relu',name="conv8"))
    model.add(Conv1D(8, 3, activation='relu',name="conv9"))
    model.add(MaxPooling1D(3,name="pool1"))
    model.add(Flatten(name="flatten"))
    model.add(Dense(1,kernel_initializer='normal',name="dense1"))

    model.compile(optimizer="adam",loss="mse")
    model.summary()
    return model

# CNN2
def CNN2(input_num = 31):
    model = Sequential()
    model.add(Conv1D(32, kernel_size=3, activation='relu',input_shape=(input_num,1),name="conv0"))
    model.add(Conv1D(64, 3, activation='relu',name="conv1"))
    model.add(Conv1D(128, 3, activation='relu',name="conv2"))
    model.add(Conv1D(64, 3, activation='relu',name="conv4"))
    model.add(Flatten())
    model.add(Dense(2048, kernel_initializer='normal', activation='relu',name="dense0"))
    model.add(Dense(1024, kernel_initializer='normal', activation='relu',name="dense1"))
    model.add(Dense(512, kernel_initializer='normal', activation='relu',name="dense2"))
    model.add(Dense(256, kernel_initializer='normal', activation='relu',name="dense3"))
    model.add(Dense(128,kernel_initializer='normal', activation='relu',name="dense5"))
    model.add(Dense(64,kernel_initializer='normal', activation='relu',name="dense6"))
    model.add(Dense(16,kernel_initializer='normal', activation='relu',name="dense7"))
    model.add(Dense(1,kernel_initializer='normal',name="dense8"))

    model.compile(optimizer="adam",loss="mse")
    model.summary()
    return model

# CNN3
def CNN3(input_num = 31):
    model = Sequential()
    model.add(Conv1D(16, kernel_size=3, activation='relu',input_shape=(input_num,1),name="conv0"))
    model.add(Conv1D(64, 3, activation='relu',name="conv1"))
    model.add(Conv1D(128, 3, activation='relu',name="conv2"))
    model.add(Conv1D(256,3, activation='relu',name="conv3"))
    model.add(Conv1D(512,3, activation='relu',name="conv4"))
    model.add(Conv1D(256, 3, activation='relu',name="conv5"))
    model.add(Conv1D(128, 3, activation='relu',name="conv6"))
    model.add(Conv1D(64, 3, activation='relu',name="conv7"))
    model.add(Conv1D(16, 3, activation='relu',name="conv8"))
    model.add(Conv1D(8, 3, activation='relu',name="conv9"))
    model.add(Conv1D(4, 3, activation='relu',name="conv10"))
    model.add(Flatten(name="flatten"))
    model.add(Dense(1,kernel_initializer='normal',name="dense1"))

    model.compile(optimizer="adam",loss="mse")
    model.summary()
    return model

# CNN4
def CNN4(input_num = 31):
    model = Sequential()
    model.add(Conv1D(16, kernel_size=3, activation='relu',input_shape=(input_num,1),name="conv0"))
    model.add(Conv1D(64, 3, activation='relu',name="conv1"))
    model.add(Conv1D(128, 3, activation='relu',name="conv2"))
    model.add(Flatten(name="flatten"))
    model.add(Dense(2048, kernel_initializer='normal', activation='relu',name="dense0"))
    model.add(Dense(1024, kernel_initializer='normal', activation='relu',name="dense1"))
    model.add(Dense(512, kernel_initializer='normal', activation='relu',name="dense2"))
    model.add(Dense(256, kernel_initializer='normal', activation='relu',name="dense3"))
    model.add(Dense(128,kernel_initializer='normal', activation='relu',name="dense5"))
    model.add(Dense(64,kernel_initializer='normal', activation='relu',name="dense6"))
    model.add(Dense(32,kernel_initializer='normal', activation='relu',name="dense7"))
    model.add(Dense(16,kernel_initializer='normal', activation='relu',name="dense8"))
    model.add(Dense(1,kernel_initializer='normal',name="dense9"))

    model.compile(optimizer="adam",loss="mse")
    model.summary()
    return model

# CNN5
def CNN5(input_num = 31):
    model = Sequential()
    model.add(Conv1D(32, kernel_size=3, activation='relu',input_shape=(input_num,1),name="conv0"))
    model.add(Conv1D(64, 3, activation='relu',name="conv1"))
    model.add(Conv1D(128, 3, activation='relu',name="conv2"))
    model.add(Conv1D(64, 3, activation='relu',name="conv4"))
    model.add(Flatten())
    model.add(Dense(2048, kernel_initializer='normal', activation='relu',name="dense0"))
    model.add(Dense(1536, kernel_initializer='normal', activation='relu',name="dense1"))
    model.add(Dense(768, kernel_initializer='normal', activation='relu',name="dense2"))
    model.add(Dense(384, kernel_initializer='normal', activation='relu',name="dense3"))
    model.add(Dense(128,kernel_initializer='normal', activation='relu',name="dense5"))
    model.add(Dense(64,kernel_initializer='normal', activation='relu',name="dense6"))
    model.add(Dense(16,kernel_initializer='normal', activation='relu',name="dense7"))
    model.add(Dense(1,kernel_initializer='normal',name="dense8"))

    model.compile(optimizer="adam",loss="mse")
    model.summary()
    return model

# CNN6
def CNN6(input_num = 31):
    model = Sequential()
    model.add(Conv1D(16, kernel_size=3, activation='relu',input_shape=(input_num,1),name="conv0"))
    model.add(Conv1D(64, 3, activation='relu',name="conv1"))
    model.add(Conv1D(128, 3, activation='relu',name="conv2"))
    model.add(Conv1D(256,3, activation='relu',name="conv3"))
    model.add(Conv1D(512,3, activation='relu',name="conv4"))
    model.add(Conv1D(256, 3, activation='relu',name="conv5"))
    model.add(Conv1D(128, 3, activation='relu',name="conv6"))
    model.add(Conv1D(64, 3, activation='relu',name="conv7"))
    model.add(Conv1D(16, 3, activation='relu',name="conv9"))
    model.add(MaxPooling1D(3,name="pool1")) 
    model.add(Flatten(name="flatten"))
    model.add(Dense(32,kernel_initializer='normal',name="dense0"))
    model.add(Dense(1,kernel_initializer='normal',name="dense1"))

    model.compile(optimizer="adam",loss="mse")
    print(model.summary())
    return model

# CNN7
def CNN7(input_num = 31):
    model = Sequential()
    model.add(Conv1D(32, kernel_size=3, activation='relu',input_shape=(input_num,1),name="conv0"))
    model.add(Conv1D(64, 3, activation='relu',name="conv1"))
    model.add(Conv1D(128, 3, activation='relu',name="conv2"))
    model.add(Conv1D(256,3, activation='relu',name="conv3"))
    model.add(Conv1D(512,3, activation='relu',name="conv4"))
    model.add(Conv1D(256, 3, activation='relu',name="conv5"))
    model.add(Conv1D(128, 3, activation='relu',name="conv6"))
    model.add(Conv1D(64, 3, activation='relu',name="conv7"))
    model.add(Conv1D(32, 3, activation='relu',name="conv8"))
    model.add(Conv1D(16, 3, activation='relu',name="conv9"))
    model.add(Flatten(name="flatten"))
    model.add(Dense(1,kernel_initializer='normal',name="dense1"))

    model.compile(optimizer="adam",loss="mse")
    model.summary()
    return model

# LSTM1
def LSTM1(input_num = 31):
    model = Sequential()
    model.add(LSTM(2048,  activation='relu', input_shape=(1, input_num), return_sequences=True))
    model.add(LSTM(2048,  activation='relu', return_sequences=True))
    #model.add(Dropout(0.2))
    model.add(LSTM(units =1024, activation='relu', return_sequences=True))
    model.add(LSTM(units =1024, activation='relu', return_sequences=True))
    #model.add(Dropout(0.2))
    model.add(LSTM(units = 512, activation='relu', return_sequences=True))
    model.add(LSTM(units = 256, activation='relu', return_sequences=True))
    #model.add(Dropout(0.2))
    model.add(LSTM(units = 128, activation='relu', return_sequences=True))
    #model.add(Dropout(0.2))`
    model.add(LSTM(units = 64, activation='relu', return_sequences=True))
    model.add(LSTM(units = 32, activation='relu', return_sequences=True))
    model.add(LSTM(units = 16, activation='relu', return_sequences=True))
    model.add(LSTM(units = 8, activation='relu', return_sequences=True))
    #model.add(LSTM(units = 8, activation='relu', return_sequences=True))
    model.add(TimeDistributed(Dense(1)))
    model.add(Flatten())

    model.compile(optimizer="adam",loss="mse")
    model.summary()
    return model

# LSTM2
def LSTM2(input_num = 31):
    model = Sequential()
    model.add(LSTM(1024,  activation='relu', input_shape=(1, input_num), return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(units =512, activation='relu', return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(units = 256, activation='relu', return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(units = 128, activation='relu', return_sequences=True))
    model.add(LSTM(units = 64, activation='relu', return_sequences=True))
    model.add(LSTM(units = 32, activation='relu', return_sequences=True))
    model.add(LSTM(units = 16, activation='relu', return_sequences=True))
    model.add(LSTM(units = 8, activation='relu', return_sequences=True))
    model.add(TimeDistributed(Dense(1)))
    model.add(Flatten())

    model.compile(optimizer="adam",loss="mse")
    model.summary()
    return model

# LSTM3
def LSTM3(input_num = 31):
    model = Sequential()
    model.add(LSTM(2048,  activation='relu', input_shape=(1, input_num), return_sequences=True))
    #model.add(LSTM(2048,  activation='relu', return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(units =1024, activation='relu', return_sequences=True))
    #model.add(LSTM(units =1024, activation='relu', return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(units = 512, activation='relu', return_sequences=True))
    model.add(LSTM(units = 256, activation='relu', return_sequences=True))
    #model.add(Dropout(0.2))
    model.add(LSTM(units = 128, activation='relu', return_sequences=True))
    #model.add(Dropout(0.2))`
    model.add(LSTM(units = 64, activation='relu', return_sequences=True))
    model.add(LSTM(units = 32, activation='relu', return_sequences=True))
    model.add(LSTM(units = 16, activation='relu', return_sequences=True))
    model.add(LSTM(units = 8, activation='relu', return_sequences=True))
    #model.add(LSTM(units = 8, activation='relu', return_sequences=True))
    model.add(TimeDistributed(Dense(1)))
    model.add(Flatten())

    model.compile(optimizer="adam",loss="mse")
    model.summary()
    return model

# LSTM4
def LSTM4(input_num = 31):
    model = Sequential()
    model.add(LSTM(2048,  activation='relu', input_shape=(1, input_num), return_sequences=True))
    model.add(LSTM(1024,  activation='relu', return_sequences=True))
    model.add(LSTM(1024,  activation='relu', return_sequences=True))
    model.add(LSTM(units =512, activation='relu', return_sequences=True))
    model.add(LSTM(units =512, activation='relu', return_sequences=True))
    model.add(LSTM(units = 256, activation='relu', return_sequences=True))
    model.add(LSTM(units = 128, activation='relu', return_sequences=True))
    model.add(LSTM(units = 64, activation='relu', return_sequences=True))
    model.add(LSTM(units = 32, activation='relu', return_sequences=True))
    model.add(LSTM(units = 16, activation='relu', return_sequences=True))
    model.add(LSTM(units = 8, activation='relu', return_sequences=True))
    model.add(TimeDistributed(Dense(1)))
    model.add(Flatten())

    model.compile(optimizer="adam",loss="mse")
    model.summary()
    return model


