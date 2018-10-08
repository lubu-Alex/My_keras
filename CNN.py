import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau
from keras.callbacks import TensorBoard
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
import os
import warnings
# 忽略硬件加速的警告信息
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings('ignore')

#导入训练和测试数据
train = np.loadtxt('train.txt')
X_train = train[:,0:len(train[0])-1].reshape(-1,28,28,1)
y_train = keras.utils.to_categorical(train[:,len(train[0])-1]-1)
test = np.loadtxt('test.txt')
X_test = test[:,0:len(test[0])-1].reshape(-1,28,28,1)
y_test = keras.utils.to_categorical(test[:,len(test[0])-1]-1)

batch_size=28
epochs=8

#建立模型
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', padding='same', data_format='channels_last',name='layer1_con1',input_shape=(28, 28, 1)))
model.add(Conv2D(32, (3, 3), activation='relu', padding='same', data_format='channels_last',name='layer1_con2'))
model.add(MaxPooling2D(pool_size=(2, 2),strides=(2,2), padding = 'same', data_format='channels_last',name = 'layer1_pool'))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), activation='relu', padding='same', data_format='channels_last',name='layer2_con1'))
model.add(Conv2D(64, (3, 3), activation='relu', padding='same', data_format='channels_last',name='layer2_con2'))
model.add(MaxPooling2D(pool_size=(2, 2),strides=(2,2), padding = 'same', data_format='channels_last',name = 'layer2_pool'))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(9, activation='softmax'))

#定义损失值、优化器
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics = ['accuracy'])

print('------------ Start Training ------------')
learning_rate_reduction = ReduceLROnPlateau(monitor = 'val_acc', patience = 3,mode='auto',
                                            verbose = 1, factor=0.5, min_lr = 0.00001)
#TensorBoard可视化
TensorBoard=TensorBoard(log_dir='./log', write_images=1, histogram_freq=1)

#保存训练参数
Checkpoint = ModelCheckpoint(filepath='./cnn_model',monitor='val_acc',mode='auto' ,save_best_only=True)

#图片增强
data_augment = ImageDataGenerator(rotation_range= 10,zoom_range= 0.1,
                                  width_shift_range = 0.1,height_shift_range = 0.1,
                                  horizontal_flip = False, vertical_flip = False)
#模型训练
model.fit_generator(data_augment.flow(X_train, y_train, batch_size=batch_size),
                             epochs= epochs, validation_data = (X_test,y_test),
                             callbacks=[learning_rate_reduction,TensorBoard,Checkpoint],shuffle=True)

#模型评估
[loss,accuracy] = model.evaluate(X_test, y_test)
print('\nTest Loss: ', loss)
print('\nTest Accuracy: ', accuracy)

