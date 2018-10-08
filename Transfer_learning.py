from keras.applications.inception_v3 import InceptionV3
from keras.models import Model
from keras.layers import GlobalAveragePooling2D
from keras.optimizers import SGD
import numpy as np
import keras
from keras.layers import Dense
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau
from keras.callbacks import TensorBoard
from keras.callbacks import ModelCheckpoint
import os
import warnings
# 忽略硬件加速的警告信息
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings('ignore')

#导入训练和测试数据
width, height, channel = 299, 299, 3
train = np.loadtxt('train_transfer.txt')
X_train = train[:,0:len(train[0])-1].reshape(-1,width,height,channel)
y_train = keras.utils.to_categorical(train[:,len(train[0])-1]-1)
test = np.loadtxt('test_transfer.txt')
X_test = test[:,0:len(test[0])-1].reshape(-1,width,height,channel)
y_test = keras.utils.to_categorical(test[:,len(test[0])-1]-1)

batch_size=28
epochs=1
Classes= 9
FC_SIZE = 128                  # 全连接层的节点个数
NB_IV3_LAYERS_TO_FREEZE = 172  # 冻结层的数量

#增加新的全连接层
def add_fully_connected_layer(base_model, classes):
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(FC_SIZE, activation='relu')(x)
    predictions = Dense(classes, activation='softmax')(x)
    model = Model(input=base_model.input, output=predictions)
    return model

#冻结base_model所有层，这样就可以正确获得bottleneck特征
def setup_to_finetune(model):
    for layer in model.layers[:NB_IV3_LAYERS_TO_FREEZE]:
     layer.trainable = False
    for layer in model.layers[NB_IV3_LAYERS_TO_FREEZE:]:
     layer.trainable = True
    model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy', metrics=['accuracy'])


#定义网络框架
base_model = InceptionV3(weights='imagenet', include_top=False)     #InceptionV3
model = add_fully_connected_layer(base_model, Classes)
setup_to_finetune(model)

print('------------ Start Training ------------')
learning_rate_reduction = ReduceLROnPlateau(monitor = 'val_acc', patience = 3,mode='auto',
                                            verbose = 1, factor=0.5, min_lr = 0.00001)
#TensorBoard可视化
TensorBoard=TensorBoard(log_dir='./Transfer_learning_log', write_images=1, histogram_freq=1)

#保存训练参数
Checkpoint = ModelCheckpoint(filepath='./Transfer_learning_model',monitor='val_acc',mode='auto' ,save_best_only=True)

#图片增强
data_augment = ImageDataGenerator(rotation_range= 10,zoom_range= 0.1,
                                  width_shift_range = 0.1,height_shift_range = 0.1,
                                  horizontal_flip = False, vertical_flip = False)

#模型训练
model.fit_generator(data_augment.flow(X_train, y_train, batch_size=batch_size),
                             epochs= epochs, validation_data = (X_test,y_test),
                             verbose =5,callbacks=[learning_rate_reduction,TensorBoard,Checkpoint],class_weight='auto',shuffle=True)

#模型评估
[loss,accuracy] = model.evaluate(X_test, y_test)
print('\nTest Loss: ', loss)
print('\nTest Accuracy: ', accuracy)