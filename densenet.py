#!/usr/bin/env python
# coding: utf-8

# In[1]:
import cv2
import numpy as np
from keras import layers
from keras.applications import DenseNet121
from keras.callbacks import Callback
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.optimizers import Adam
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import cohen_kappa_score
import tensorflow as tf
from tqdm import tqdm

#get_ipython().run_line_magic('matplotlib', 'inline')
import warnings
warnings.filterwarnings("ignore")


# In[2]:


np.random.seed(2019)
tf.set_random_seed(2019)


# In[3]:


train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')
print(train_df.shape)
print(test_df.shape)
train_df.head()


# In[4]:


train_df['diagnosis'].hist()
train_df['diagnosis'].value_counts()


# In[5]:


def preprocess_image(image_path, desired_size=224):
    #im = Image.open(image_path)
    #im = im.resize((desired_size, )*2, resample=Image.LANCZOS)
    im = cv2.imread(image_path)
    im = cv2.cvtColor(im,cv2.COLOR_BGR2RGB)
    im = cv2.resize(im,(desired_size,desired_size),interpolation=cv2.INTER_LANCZOS4)
    im = cv2.addWeighted(im,4,cv2.GaussianBlur(im,(0,0),256/30),-4,128)
    return im


# In[6]:


N = train_df.shape[0]
x_train = np.empty((N, 224, 224, 3), dtype=np.uint8)

for i, image_id in enumerate(tqdm(train_df['id_code'])):
    x_train[i, :, :, :] = preprocess_image(
        f'train_images/{image_id}.png'
    )


# In[ ]:


# N = test_df.shape[0]
# x_test = np.empty((N, 224, 224, 3), dtype=np.uint8)

# for i, image_id in enumerate(tqdm(test_df['id_code'])):
#     x_test[i, :, :, :] = preprocess_image(
#         f'test_images/{image_id}.png'
#     )


# In[7]:


y_train = pd.get_dummies(train_df['diagnosis']).values

print(x_train.shape)
print(y_train.shape)
# print(x_test.shape)


# In[8]:


y_train_multi = np.empty(y_train.shape, dtype=y_train.dtype)
y_train_multi[:, 4] = y_train[:, 4]

for i in range(3, -1, -1):
    y_train_multi[:, i] = np.logical_or(y_train[:, i], y_train_multi[:, i+1])

print("Original y_train:", y_train.sum(axis=0))
print("Multilabel version:", y_train_multi.sum(axis=0))


# In[9]:


x_train, x_val, y_train, y_val = train_test_split(
    x_train, y_train_multi,
    test_size=0.15,
    random_state=2019
)


# In[10]:


BATCH_SIZE = 32

def create_datagen():
    return ImageDataGenerator(
        zoom_range=0.15,  # set range for random zoom
        # set mode for filling points outside the input boundaries
        fill_mode='constant',
        cval=0.,  # value used for fill_mode = "constant"
        horizontal_flip=True,  # randomly flip images
        vertical_flip=True,  # randomly flip images
    )

# Using original generator
data_generator = create_datagen().flow(x_train, y_train, batch_size=BATCH_SIZE, seed=2019)


# In[11]:


class Metrics(Callback):
    def on_train_begin(self):
        self.val_kappas = []

    def on_epoch_end(self, epoch):
        X_val, y_val = self.validation_data[:2]
        y_val = y_val.sum(axis=1) - 1
        
        y_pred = self.model.predict(X_val) > 0.5
        y_pred = y_pred.astype(int).sum(axis=1) - 1

        _val_kappa = cohen_kappa_score(
            y_val,
            y_pred,
            weights='quadratic'
        )

        self.val_kappas.append(_val_kappa)

        print(f"val_kappa: {_val_kappa:.4f}")
        
        if _val_kappa == max(self.val_kappas):
            print("Validation Kappa has improved. Saving model.")
            self.model.save('model_224_gaussian_blur_512_256.h5')

        return


# In[12]:


densenet = DenseNet121(
    weights='DenseNet-BC-121-32-no-top.h5',
    include_top=False,
    input_shape=(224,224,3)
)


# In[15]:


def build_model():
    model = Sequential()
    model.add(densenet)
    model.add(layers.GlobalAveragePooling2D())
    model.add(layers.Dense(512,activation = 'relu'))
    model.add(layers.Dropout(0.3))
    model.add(layers.Dense(256,activation = 'relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(5, activation='sigmoid'))
    
    model.compile(
        loss='binary_crossentropy',
        optimizer=Adam(lr=0.00005),
        metrics=['accuracy']
    )
    return model


# In[16]:


model = build_model()
model.summary()


# In[17]:


kappa_metrics = Metrics()

history = model.fit_generator(
    data_generator,
    steps_per_epoch=x_train.shape[0] / BATCH_SIZE,
    epochs=15,
    validation_data=(x_val, y_val),
    callbacks=[kappa_metrics]
)


# In[ ]:


# y_test = model.predict(x_test) > 0.5
# y_test = y_test.astype(int).sum(axis=1) - 1

# test_df['diagnosis'] = y_test
# test_df.to_csv('submission.csv',index=False)


# In[ ]:




