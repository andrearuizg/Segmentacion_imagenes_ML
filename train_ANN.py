import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.metrics import Recall, Precision
from tensorflow.keras.callbacks import (EarlyStopping, ModelCheckpoint, 
ReduceLROnPlateau, CSVLogger, TensorBoard)
from tensorflow.keras.optimizers import Adam

train = pd.read_csv(r"files/data.txt",sep=',')

print(len(train))

predictors = ['r', 't', 'B', 'G', 'R']
X = train[predictors].values
y = train['m'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, test_size=0.2, shuffle=False)

scaler = StandardScaler()# Ejercicio, no use la escalización de los datos a ver que tal funciona!
scaler.fit(X_train)# el fit de los datos solo se hace con el conjunto de entrenamiento!
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
y_train = y_train.reshape(-1, 1)
y_test = y_test.reshape(-1, 1)
print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)

train_data = tf.data.Dataset.from_tensor_slices((X_train, y_train))
valid_data = tf.data.Dataset.from_tensor_slices((X_test, y_test))

batch_ = 10000
epochs_ = int(len(X_train) / batch_)

train_data = train_data.batch(batch_)
valid_data = valid_data.batch(batch_)

options = tf.data.Options()
options.experimental_distribute.auto_shard_policy = (tf.data.experimental.AutoShardPolicy.OFF)

train_data = train_data.with_options(options)
valid_data = valid_data.with_options(options)

strategy = tf.distribute.MirroredStrategy()

print('Number of devices: {}'.format(strategy.num_replicas_in_sync))

with strategy.scope():
    nn_model = Sequential()
    nn_model.add(Dense(10, input_dim = 5, activation='relu'))
    nn_model.add(Dense(15, activation='relu'))
    nn_model.add(Dense(25, activation='relu'))
    nn_model.add(Dense(125, activation='relu'))
    nn_model.add(Dense(625, activation='relu'))
    nn_model.add(Dense(125, activation='relu'))
    nn_model.add(Dense(25, activation='relu'))
    nn_model.add(Dense(15, activation='relu'))
    nn_model.add(Dense(10, activation='relu'))
    nn_model.add(Dense(1, activation='sigmoid'))
    with open('files/model_summary.txt', 'w') as fh:
        nn_model.summary(print_fn=lambda x: fh.write(x + '\n'))
    nn_model.compile(loss='mean_squared_error', optimizer='adam', metrics=["acc", Precision(), Recall()])
    early_stop = EarlyStopping(monitor='val_loss', patience=16, verbose=1)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, min_lr=0.001)

    nn_model.fit(train_data, epochs=epochs_, validation_data=valid_data, batch_size=batch_,callbacks=[early_stop, reduce_lr], shuffle=True)

nn_model.save('files/model.h5')
print('Terminó °_°')

