#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
from tqdm import tqdm_notebook
from progress.bar import Bar
from sklearn.model_selection import train_test_split
import os,cv2
from keras.layers import Dropout,Dense,GlobalAveragePooling2D
from keras.models import Model
from keras.applications.vgg19 import VGG19
from keras.applications.densenet import DenseNet121
from keras.applications.inception_v3 import InceptionV3
from keras.applications.mobilenet import MobileNet
from keras.applications.xception import Xception
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam,RMSprop
from keras.utils import to_categorical


import warnings
warnings.filterwarnings("ignore")

train_csv = pd.read_csv('train.csv')

train_path = 'train_images/'
test_path = 'test_images/'


print("Total number of images in No DR class: ",len(train_csv[train_csv['diagnosis'] == 0]))
print("Total number of images in Mild class: ",len(train_csv[train_csv['diagnosis'] == 1]))
print("Total number of images in Moderate class: ",len(train_csv[train_csv['diagnosis'] == 2]))
print("Total number of images in Severe class: ",len(train_csv[train_csv['diagnosis'] == 3]))
print("Total number of images in Proliferative DR class: ",len(train_csv[train_csv['diagnosis'] == 4]))
print("Total number of train images: ",len(train_csv))


def get_data(img_path,image_csv):
    X = []
    Y = []
    images_list = os.listdir(img_path)
    bar = Bar('Processing',max = len(images_list))
    for i in tqdm_notebook(range(len(images_list))):
        img = cv2.imread(img_path + image_csv['id_code'][i] + '.png')
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
#        img = cv2.addWeighted(img,4,cv2.GaussianBlur(img,(0,0),256/30),-4,128)
        X.append(cv2.resize(img,(256,256)))
        Y.append(image_csv['diagnosis'][i])
        bar.next()
    bar.finish()
    return np.array(X),np.array(Y)

train_X,train_Y = get_data(train_path,train_csv)
print('train_X max value: ',train_X.max())
train_X = train_X/255.0
print('train_X max value: ',train_X.max())

def pretrained_model(model,img_shape,trainable = False,weights = None,optim='adam',lr = 0.001,weighted_metrics = None):
    if model == 'densenet121':
        base_model = DenseNet121(include_top=False,weights= weights,input_shape = img_shape)
    elif model == 'inception':
        base_model = InceptionV3(include_top=False,weights= weights,input_shape = img_shape)
    elif model == 'mobilenet':
        base_model = MobileNet(include_top=False,weights= weights,input_shape = img_shape)
    elif model == 'vgg':
        base_model = VGG19(include_top=False,weights= weights,input_shape = img_shape)
    elif model == 'resnet50':
        base_model = ResNet50V2(include_top=False,weights= weights,input_shape = img_shape)
    elif model =='resnet101':
        base_model = ResNet101V2(include_top = False,weights = weights,input_shape = img_shape)
    elif model == 'xception':
        base_model = Xception(include_top=False,weights= weights,input_shape = img_shape)
    for layer in base_model.layers:
        layer.trainable = trainable
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.3)(x)
    x = Dense(2048,activation='relu')(x)
    x = Dense(1024,activation='relu')(x)
    x = Dropout(0.25)(x)
    x = Dense(512,activation = 'relu')(x)
    x = Dense(128,activation='relu')(x)
    x = Dropout(0.25)(x)
    predictions = Dense(2,activation='softmax')(x)
    model = Model(base_model.input,predictions)
    print(model.summary())

    if optim == 'adam':
        model.compile(optimizer=Adam(lr),loss = 'categorical_crossentropy',
                      metrics=['accuracy'],weighted_metrics=weighted_metrics)
    elif optim == 'rms':
        model.compile(optimizer=RMSprop(lr),loss = 'categorical_crossentropy',
                      metrics=['accuracy'],weighted_metrics= weighted_metrics)
    return model


model = pretrained_model('densenet121',(256,256,3),weights = None,lr = 0.0001)
print("model created")
#csv_logger = CSVLogger("result/csv/dense_tr_2.csv",separator = ",",append=True)
checkpoint = ModelCheckpoint('result/model/dense_tr_binary_2.h5',monitor='val_acc',
                         verbose=1,
                        save_best_only= True)

#learning_rate = ReduceLROnPlateau(monitor='val_acc',
#                             factor = 0.1,
#                             patience = 2,
#                             verbose = 1)
#callback = [checkpoint,learning_rate,csv_logger]

X_tr,X_va,Y_tr,Y_va = train_test_split(train_X,train_Y,random_state = 42,shuffle = True,
                                       test_size = 0.20,stratify = train_Y)
del train_X,train_Y
np.save('X_va',X_va)
np.save('Y_va',Y_va)
def to_binary(data):
	for i,_ in enumerate(data):
		if data[i] != 0:
			data[i] = 1
	return np.array(data)

Y_tr = to_binary(Y_tr)
Y_va = to_binary(Y_va)

print(Y_tr.shape)
print(Y_va.shape)
Y_tr = to_categorical(Y_tr)
Y_va = to_categorical(Y_va)

print('train_X shape: ',X_tr.shape)
print('train_Y shape: ',Y_tr.shape)
print('X_valid shape: ',X_va.shape)
print('Y_valid shape: ',Y_va.shape)
print('train_Y: \n',Y_tr[:5])
#datagen = ImageDataGenerator(rotation_range = 30,
#         		     horizontal_flip = True,
#			     vertical_flip = True,
#			     rescale = 1.0/255)

#model.fit_generator(datagen.flow(train_X,train_Y,batch_size=32),
#				steps_per_epoch = len(train_X)//32,
#				epochs=50,
#				verbose=1,
#				callbacks=callback,
#				validation_data=datagen.flow(X_va,Y_va,batch_size = 32),
#				validation_steps = len(X_va)//32)

model.fit(X_tr,Y_tr,batch_size = 32,epochs =50,verbose =1,validation_data = (X_va,Y_va),callbacks=[checkpoint])

from sklearn.metrics import confusion_matrix,accuracy_score

y_pred = model.predict(X_va,verbose=1)
y_pred = np.argmax(y_pred,axis = 1)
print(y_pred.shape)

y_true = np.argmax(Y_va,axis = 1)
print("accuracy_score: ",accuracy_score(y_true,y_pred))
print("confusion_matrix: ",confusion_matrix(y_true,y_pred))
#np.save('confusion_matrix_5',confusion_matrix(y_true,y_pred))
#model.save_weights('result/model/dense_tr_binary.h5')
