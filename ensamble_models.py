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


train_files = os.listdir('/home/speri/precipitation/train')
train = []
for file in train_files:
    try:
        data = np.load('/home/speri/precipitation/train/'+file).astype('float32')
        train.append(data)
    except:
        continue

submission = pd.read_csv('/home/speri/precipitation/sample_submission.csv')
test = []
for sub_id in submission['id']:
    data = np.load('/home/speri/precipitation/test/'+'subset_'+sub_id+'.npy').astype('float32')
    test.append(data)

train = np.array(train)
test = np.array(test)

x_train = train[:,:,:,:10]
y_train = train[:,:,:,14]
test = test[:,:,:,:10]
del train

x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.025, random_state=7777)
print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)

y_train_ = y_train.reshape(-1,y_train.shape[1]*y_train.shape[2])
x_train = np.delete(x_train, np.where(y_train_<0)[0], axis=0)
y_train = np.delete(y_train, np.where(y_train_<0)[0], axis=0)
y_train = y_train.reshape(-1, x_train.shape[1], x_train.shape[2],1)
y_test = y_test.reshape(-1, y_test.shape[1], y_test.shape[2],1)
print(x_train.shape, y_train.shape)


def data_generator(x_train, y_train):
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

    return x_train, y_train


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

def create_model():
    print('Creating model')
    inputs=Input(x_train.shape[1:])

    bn=BatchNormalization()(inputs)
    conv0=Conv2D(256, kernel_size=1, kernel_initializer='he_uniform', bias_initializer='zeros', strides=1, padding='same', activation='relu')(bn)

    bn=BatchNormalization()(conv0)
    conv=Conv2D(128, kernel_size=2, kernel_initializer='he_uniform', bias_initializer='zeros',strides=1, padding='same', activation='relu')(bn)
    concat=concatenate([conv0, conv], axis=3)

    bn=BatchNormalization()(concat)
    conv=Conv2D(64, kernel_size=3,kernel_initializer='he_uniform', bias_initializer='zeros', strides=1, padding='same', activation='relu')(bn)
    concat=concatenate([concat, conv], axis=3)

    for i in range(5):
        bn=BatchNormalization()(concat)
        conv=Conv2D(32, kernel_size=3,kernel_initializer='he_uniform', bias_initializer='zeros', strides=1, padding='same', activation='relu')(bn)
        concat=concatenate([concat, conv], axis=3)

    bn=BatchNormalization()(concat)
    outputs=Conv2D(1, kernel_size=1,kernel_initializer='he_uniform', bias_initializer='zeros', strides=1, padding='same', activation='relu')(bn)

    model=Model(inputs=inputs, outputs=outputs)
    print('Model created successfully')
    return model

def train_model(x_data, y_data, k, s):
    print('Training Model')
    k_fold = KFold(n_splits=k, shuffle=True, random_state=7777)
    model_number = 0
    for train_idx, val_idx in k_fold.split(x_data):
        if model_number == s:
            x_train, y_train = x_data[train_idx], y_data[train_idx]
            x_val, y_val = x_data[val_idx], y_data[val_idx]
            x_train, y_train = data_generator(x_train, y_train)
            model = create_model()
            model.compile(loss='mae', optimizer='adam', metrics=[score, fscore_keras])
            callbacks_list = [tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss',patience=3,factor=0.8),tf.keras.callbacks.ModelCheckpoint(filepath = '/home/speri/precipitation/models/model'+str(model_number)+'.h5',monitor='val_score',save_best_only=True)]

            model.fit(x_train, y_train, epochs=25, batch_size=64, validation_data=(x_val, y_val), callbacks=callbacks_list)

        model_number+=1

k = 4
models = []

train_model(x_train, y_train, k=k, s=4)
print('Trianing completed')
for n in range(k):
    model = load_model('/home/speri/precipitation/models/model'+str(n)+'.h5', custom_objects = {'score':score,'fscore_keras':fscore_keras})
    models.append(model)


preds = []
for model in models:
    preds.append(model.predict(x_test))
    print(mae_over_fscore(y_test, preds[-1]))

pred = sum(preds)/len(preds)
print(mae_over_fscore(y_test, pred))

preds = []
for model in models:
    preds.append(model.predict(test))

pred = sum(preds)/len(preds)


submission.iloc[:,1:] = pred.reshape(-1,1600)
print('submission:', submission)

submission.to_csv('submission.csv', index=False)
