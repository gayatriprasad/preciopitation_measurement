# import required libraries
import numpy as np
import pandas as pd
import os
from sklearn.metrics import f1_score
import tensorflow as tf
from tensorflow.keras.datasets.mnist import load_data
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.layers import Input, Conv2D, Add, BatchNormalization, concatenate
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

### read data and selecting the relevant features
# get the train data
train_files = os.listdir('/home/speri/precipitation/train')
train = []
for file in train_files:
    try:
        data = np.load('/home/speri/precipitation/train/'+file).astype('float32')
        train.append(data)
    except:
        continue

# get the train data
submission = pd.read_csv('/home/speri/precipitation/sample_submission.csv')
test = []
for sub_id in submission['id']:
    data = np.load('/home/speri/precipitation/test/'+'subset_'+sub_id+'.npy').astype('float32')
    test.append(data)

# convert to numpy arrays
train = np.array(train)
print('train_shape :', train.shape)
test = np.array(test)
print('test_shape :', test.shape)

# extracting channels 0-10 for learning
x_train = train[:,:,:,:10]
y_train = train[:,:,:,14]
test = test[:,:,:,:10]
del train

### preparing the data for training
# spliting train data for th
x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.025, random_state=7777)
print('x_train.shape, y_train.shape, x_test.shape, y_test.shape :', x_train.shape, y_train.shape, x_test.shape, y_test.shape)

# deleting data containing values <200b><200b>less than 0
y_train_ = y_train.reshape(-1,y_train.shape[1]*y_train.shape[2])
x_train = np.delete(x_train, np.where(y_train_<0)[0], axis=0)
y_train = np.delete(y_train, np.where(y_train_<0)[0], axis=0)
y_train = y_train.reshape(-1, x_train.shape[1], x_train.shape[2],1)
y_test = y_test.reshape(-1, y_test.shape[1], y_test.shape[2],1)
print('x_train.shape, y_train.shape: ', x_train.shape, y_train.shape)

###data generator
def data_generator(x_train, y_train):

    print('Generating data -- data augmentation -- starts')

    rotate_X_90 = np.zeros_like(x_train)
    rotate_Y_90 = np.zeros_like(y_train)

    for j in range(rotate_X_90.shape[0]):
        rotate_x=np.zeros([x_train.shape[1],x_train.shape[2],10])
        rotate_y=np.zeros([x_train.shape[1],x_train.shape[2],1])
        for i in range(10):
            rotate_x[:,:,i]=np.rot90(x_train[j,:,:,i])
        rotate_y[:,:,0]=np.rot90(y_train[j,:,:,0])

        rotate_X_90[j,:,:,:] = rotate_x
        rotate_Y_90[j,:,:,:] = rotate_y

    rotate_X_180 = np.zeros_like(x_train)
    rotate_Y_180 = np.zeros_like(y_train)

    for j in range(rotate_X_180.shape[0]):
        rotate_x=np.zeros([x_train.shape[1],x_train.shape[2],10])
        rotate_y=np.zeros([x_train.shape[1],x_train.shape[2],1])
        for i in range(10):
            rotate_x[:,:,i]=np.rot90(x_train[j,:,:,i])
            rotate_x[:,:,i]=np.rot90(rotate_x[:,:,i])
        rotate_y[:,:,0]=np.rot90(y_train[j,:,:,0])
        rotate_y[:,:,0]=np.rot90(rotate_y[:,:,0])

        rotate_X_180[j,:,:,:] = rotate_x
        rotate_Y_180[j,:,:,:] = rotate_y

    rotate_X_270 = np.zeros_like(x_train)
    rotate_Y_270 = np.zeros_like(y_train)

    for j in range(rotate_X_270.shape[0]):
        rotate_x=np.zeros([x_train.shape[1],x_train.shape[2],10])
        rotate_y=np.zeros([x_train.shape[1],x_train.shape[2],1])
        for i in range(10):
            rotate_x[:,:,i]=np.rot90(x_train[j,:,:,i])
            rotate_x[:,:,i]=np.rot90(rotate_x[:,:,i])
            rotate_x[:,:,i]=np.rot90(rotate_x[:,:,i])
        rotate_y[:,:,0]=np.rot90(y_train[j,:,:,0])
        rotate_y[:,:,0]=np.rot90(rotate_y[:,:,0])
        rotate_y[:,:,0]=np.rot90(rotate_y[:,:,0])

        rotate_X_270[j,:,:,:] = rotate_x
        rotate_Y_270[j,:,:,:] = rotate_y

    x_train = np.concatenate((x_train, rotate_X_90, rotate_X_180, rotate_X_270), axis = 0)
    y_train = np.concatenate((y_train, rotate_Y_90, rotate_Y_180, rotate_Y_270), axis = 0)
    del rotate_X_90, rotate_X_180, rotate_X_270, rotate_Y_90, rotate_Y_180, rotate_Y_270


    print('Generated data -- data augmentation -- ends successfully')

    return x_train, y_train

### defining the metrics for evaluation of the model
# evaluation function
def mae_over_fscore(y_true, y_pred):
    '''
    y_true: sample_submission.csv 형태의 실제 값
    y_pred: sample_submission.csv 형태의 예측 값
    '''
    y_true = np.array(y_true)
    y_true = y_true.reshape(1, -1)[0]
    y_pred = np.array(y_pred)
    y_pred = y_pred.reshape(1, -1)[0]
    IsGreaterThanEqualTo_PointOne = y_true >= 0.1
    IsNotMissing = y_true >= 0
    mae = np.mean(np.abs(y_true[IsGreaterThanEqualTo_PointOne] - y_pred[IsGreaterThanEqualTo_PointOne]))
    y_true = np.where(y_true[IsNotMissing] >= 0.1, 1, 0)
    y_pred = np.where(y_pred[IsNotMissing] >= 0.1, 1, 0)
    f_score = f1_score(y_true, y_pred)
    return mae / (f_score + 1e-07)

def mae(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    y_true = y_true.reshape(1, -1)[0]
    y_pred = y_pred.reshape(1, -1)[0]
    over_threshold = y_true >= 0.1
    return np.mean(np.abs(y_true[over_threshold] - y_pred[over_threshold]))

def fscore(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    y_true = y_true.reshape(1, -1)[0]
    y_pred = y_pred.reshape(1, -1)[0]
    remove_NAs = y_true >= 0
    y_true = np.where(y_true[remove_NAs] >= 0.1, 1, 0)
    y_pred = np.where(y_pred[remove_NAs] >= 0.1, 1, 0)
    return(f1_score(y_true, y_pred))

def maeOverFscore(y_true, y_pred):
    return mae(y_true, y_pred) / (fscore(y_true, y_pred) + 1e-07)

def fscore_keras(y_true, y_pred):
    score = tf.py_function(func=fscore, inp=[y_true, y_pred], Tout=tf.float32, name='fscore_keras')
    return score

def score(y_true, y_pred):
    score = tf.py_function(func=maeOverFscore, inp=[y_true, y_pred], Tout=tf.float32,  name='custom_mse')
    return score

# defining a model with convolution and residual blocks   kernel_initializer='he_uniform', bias_initializer='zeros'
def create_model():

    print('Generating model -- starts')
    inputs=Input(x_train.shape[1:])
    bn=BatchNormalization()(inputs)
    conv0=Conv2D(256, kernel_size=1, kernel_initializer='he_uniform', bias_initializer='zeros', strides=1, padding='same', activation='relu')(bn)
    bn=BatchNormalization()(conv0)
    conv=Conv2D(128, kernel_size=2, kernel_initializer='he_uniform', bias_initializer='zeros', strides=1, padding='same', activation='relu')(bn)
    concat=concatenate([conv0, conv], axis=3)
    bn=BatchNormalization()(concat)
    conv=Conv2D(64, kernel_size=3, kernel_initializer='he_uniform', bias_initializer='zeros', strides=1, padding='same', activation='relu')(bn)
    concat=concatenate([concat, conv], axis=3)
    for i in range(8):
        bn=BatchNormalization()(concat)
        conv=Conv2D(32, kernel_size=3, kernel_initializer='he_uniform', bias_initializer='zeros', strides=1, padding='same', activation='relu')(bn)
        concat=concatenate([concat, conv], axis=3)

    bn=BatchNormalization()(concat)
    outputs=Conv2D(1, kernel_size=1, kernel_initializer='he_uniform', bias_initializer='zeros', strides=1, padding='same', activation='relu')(bn)
    model=Model(inputs=inputs, outputs=outputs)
    print('Generating model -- ends')
    return model

# function to train the model ---- uses k-fold cross validation
def train_model(x_data, y_data, k):

    print('model training -- begins')
    k_fold = KFold(n_splits=k, shuffle=True, random_state=7777)
    for train_idx, val_idx in k_fold.split(x_data):
        x_train, y_train = x_data[train_idx], y_data[train_idx]
        x_val, y_val = x_data[val_idx], y_data[val_idx]
        x_train, y_train = data_generator(x_train, y_train)
        model = create_model()
        model.compile(loss='mae', optimizer='adam', metrics=[score, fscore_keras])
        callbacks_list = [tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss',patience=2,factor=0.8)]
        model.fit(x_train, y_train, epochs=30, batch_size=8, validation_data=(x_val, y_val), callbacks=callbacks_list)
        print('model training -- ends')

# calling the model
k = 4
train_model(x_train, y_train, k=k)

# prediction over x_test (dataset created from splitting original data)
preds = model.predict(x_test)
print(mae_over_fscore(y_test, preds))

preds = model.predict(test)

pred = sum(preds)/len(preds)

# reshaping predictions into
submission.iloc[:,1:] = pred.reshape(-1,1600)
print(submission)

submission.to_csv('submission.csv', index = False)

print("model has finished predicting and writing to csv")
