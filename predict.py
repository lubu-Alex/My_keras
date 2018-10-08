from keras.models import load_model
import numpy as np
import glob as gb
import cv2
import os
import warnings
# 忽略硬件加速的警告信息
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings('ignore')

#加载模型
model = load_model('cnn_model')

#预测图片
def predict_img(img_path):
    img_path=gb.glob(img_path)
    for path in img_path:
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).reshape(1, 28*28) / 255.0
        img = np.array(img).astype(np.float32).reshape(-1,28,28,1)
        y_predict = model.predict_classes(img) + 1
        print('The number in picture {} is {}'.format(path,y_predict))

if __name__ == "__main__":
    predict_img(r"G:\task\script\keras_cnn\shuma_test\*.jpg")
