import keras
from keras.datasets import mnist
from keras.layers import Dense, Conv2D, MaxPool2D, Flatten, Dropout
from keras.models import Sequential, load_model
from keras.utils import to_categorical
import os
import numpy as np
import matplotlib.pyplot as plt
from keras.losses import categorical_crossentropy
from keras.optimizers import Adam, SGD, RMSprop

print('Keras Version: ', keras.__version__)

(x_train, y_train), (x_test, y_test) = mnist.load_data()

print('type(x_train) = ', type(x_train))
print('Download Prime:', x_train.shape)
np.set_printoptions(linewidth=500)
print('x_train[0] = \n', x_train[0])
print('x_test.shape = ', x_test.shape)
print('y_train = ', y_train)
print('y_train.shape = ', y_train.shape)
print('='*80)
# train_images = x_train.reshape((-1, 28*28)) / 255
# test_images = x_test.reshape((-1, 28*28)) / 255
x_train = x_train.reshape((-1, 28, 28, 1))
x_test = x_test.reshape((-1, 28, 28, 1))
print(x_train.shape)
print(x_test.shape)
print('=-'*40)
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
np.set_printoptions(edgeitems=10)
print('y_train.shape =', y_train.shape)
print('y_train = \n', y_train)


if os.path.exists('11Model.h5'):
    print('载入模型...')
    model = load_model('Model.h5')
else:
    print('训练模型...')
    model = Sequential()
    model.add(Conv2D(filters=6, kernel_size=(5,5), kernel_regularizer='l2', activation='relu', input_shape=(28,28,1)))
    model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))
    model.add(Conv2D(filters=16, kernel_size=(5,5), kernel_regularizer='l2', activation='relu'))
    model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))
    model.add(Flatten())
    model.add(Dropout(0.5))
    model.add(Dense(128, activation='relu', input_shape=(28*28, )))
    model.add(Dense(32, activation='relu', input_shape=(28*28, )))
    model.add(Dense(10, activation='softmax'))

print('模型结构：')
model.summary()

print('训练：\n')
model.compile(loss='categorical_crossentropy', optimizer=Adam(0.0001), metrics=['accuracy'])
model_log = model.fit(x_train, y_train, batch_size=50, epochs=10)
model.save('Model.h5')

test_loss, test_acc = model.evaluate(x_test, y_test)
print('测试集损失值、正确率：', test_loss, test_acc)

Loss = model_log.history['loss']
acc = model_log.history['acc']
print(Loss)
print(acc)
plt.figure()
plt.plot(acc[1:], 'r-', lw=3, label='Acc')
plt.legend(loc='lower left')
plt.figure()
plt.plot(Loss[1:], 'b-', lw=3, label='Loss')
plt.legend(loc='lower left')
plt.show()
