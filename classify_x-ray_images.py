#!/usr/bin/env python
# coding: utf-8

# In[1]:


# 安裝cv2與tensorflow套件後用這些套件建構深度學習模型
#install cv2 and tensorflow package and use the package to establish the deep learning model

import glob
import random as rn
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
import cv2
#import sklearn

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import RMSprop, Adam
from tensorflow.keras.utils import to_categorical
#from sklearn.metrics import classification_report, confusion_matrix


#get_ipython().run_line_magic('matplotlib', 'inline')
import warnings
warnings.filterwarnings('ignore')

# 下載資料load data 並分解成 訓練training, 測試test 與 驗證validation 集
# down load the data and divide it into training test and validation set
size=224
path = './chest_xray/'
# define paths
train_normal_dir = path + 'train/NORMAL/'
train_pneu_dir = path + 'train/PNEUMONIA/'

test_normal_dir = path + 'test/NORMAL/'
test_pneu_dir = path + 'test/PNEUMONIA/'

val_normal_dir = path + 'val/NORMAL/'
val_pneu_dir = path + 'val/PNEUMONIA/'


# find all files, our files have extension jpeg
train_normal_cases = glob.glob(train_normal_dir + '*jpeg')
train_pneu_cases = glob.glob(train_pneu_dir + '*jpeg')

test_normal_cases = glob.glob(test_normal_dir + '*jpeg')
test_pneu_cases = glob.glob(test_pneu_dir + '*jpeg')

val_normal_cases = glob.glob(val_normal_dir + '*jpeg')
val_pneu_cases = glob.glob(val_pneu_dir + '*jpeg')


# make path using / instead of \\ ... this may be a redudant step
train_normal_cases = [x.replace('\\', '/') for x in train_normal_cases]
train_pneu_cases = [x.replace('\\', '/') for x in train_pneu_cases]
test_normal_cases = [x.replace('\\', '/') for x in test_normal_cases]
test_pneu_cases = [x.replace('\\', '/') for x in test_pneu_cases]
val_normal_cases = [x.replace('\\', '/') for x in val_normal_cases]
val_pneu_cases = [x.replace('\\', '/') for x in val_pneu_cases]


# create lists for train, test & validation cases, create labels as well
train_list = []
test_list = []
val_list = []

for x in train_normal_cases:
    train_list.append([x, 0])
    
for x in train_pneu_cases:
    train_list.append([x, 1])
    
for x in test_normal_cases:
    test_list.append([x, 0])
    
for x in test_pneu_cases:
    test_list.append([x, 1])
    
for x in val_normal_cases:
    val_list.append([x, 0])
    
for x in val_pneu_cases:
    val_list.append([x, 1])


# shuffle/randomize data as they were loaded in order: normal cases, then pneumonia cases
rn.shuffle(train_list)
rn.shuffle(test_list)
rn.shuffle(val_list)


# create dataframes
train_df = pd.DataFrame(train_list, columns=['image', 'label'])
test_df = pd.DataFrame(test_list, columns=['image', 'label'])
val_df = pd.DataFrame(val_list, columns=['image', 'label'])
print(train_df)

# visialization of result

plt.figure(figsize=(20,5))

plt.subplot(1,3,1)
sns.countplot(train_df['label'])
plt.title('Train data')

plt.subplot(1,3,2)
sns.countplot(test_df['label'])
plt.title('Test data')

plt.subplot(1,3,3)
sns.countplot(val_df['label'])
plt.title('Validation data')

#plt.show()

# show a few cases

plt.figure(figsize=(20,8))
for i,img_path in enumerate(train_df[train_df['label'] == 1][0:4]['image']):
    plt.subplot(2,4,i+1)
    plt.axis('off')
    img = plt.imread(img_path)
    plt.imshow(img, cmap='gray')
    plt.title('Pneumonia')
#plt.show()
for i,img_path in enumerate(train_df[train_df['label'] == 0][0:4]['image']):
    plt.subplot(2,4,4+i+1)
    plt.axis('off')
    img = plt.imread(img_path)
    plt.imshow(img, cmap='gray')
    plt.title('Healthy / Normal')
#plt.show()

data = []
vdata=[]
tdata = []
y = []
vy=[]
ty = []
for index, row in train_df.iterrows():
    img = cv2.imread(row[0])
    img_resize = cv2.resize(img, (size, size))
    data.append(img_resize)
    print(index, row[0])
    y.append(row[1])
for index, row in val_df.iterrows():
    img = cv2.imread(row[0])
    img_resize = cv2.resize(img, (size, size))
    vdata.append(img_resize)
    print(index, row[0])
    vy.append(row[1])
for index, row in test_df.iterrows():
    img = cv2.imread(row[0])
    img_resize = cv2.resize(img, (size, size))
    tdata.append(img_resize)
    print(index, row[0])
    ty.append(row[1])
print(vy)
data = np.array(data)
classes = np.array(y)
np.save('data.npy', data)
np.save('classes.npy', classes)

vdata = np.array(vdata)
vclasses = np.array(vy)
np.save('vdata.npy', vdata)
np.save('vclasses.npy', vclasses)

tdata = np.array(tdata)
tclasses = np.array(ty)
np.save('tdata.npy', tdata)
np.save('tclasses.npy', tclasses)
# Shuffle the data

data = np.load('data.npy')
classes = np.load('classes.npy')
vdata = np.load('vdata.npy')
vclasses = np.load('vclasses.npy')
tdata = np.load('tdata.npy')
tclasses = np.load('tclasses.npy')


idx = np.arange(data.shape[0])
np.random.shuffle(idx)
data = data[idx]
classes = classes[idx]

idx = np.arange(vdata.shape[0])
np.random.shuffle(idx)
vdata = vdata[idx]
vclasses = vclasses[idx]

idx = np.arange(tdata.shape[0])
np.random.shuffle(idx)
tdata = tdata[idx]
tclasses = tclasses[idx]

# 1. Normalize the training and testing dataset
# 2. One hot encoding the classes

# In[12]:


X_train_norm = data / 255
X_v_norm = vdata / 255
X_t_norm = tdata / 255
#X_test_norm = X_test / 255


# In[13]:


from keras.utils import np_utils
# one-hot encoding
y_train_onehot = np_utils.to_categorical(classes, num_classes=2)
y_v_onehot = np_utils.to_categorical(vclasses, num_classes=2)
y_t_onehot = np_utils.to_categorical(tclasses, num_classes=2)
#y_test_onehot = np_utils.to_categorical(y_test, num_classes=2)

# ### 2. Training Model
# Write a CNN-model

# In[19]:

from keras.models import Sequential
from keras.layers import Dense, Activation, Conv2D, MaxPooling2D, Dropout, Flatten
model = Sequential()
#input shape is the img shape
model.add(Conv2D(filters=96, strides=(4,4), kernel_size=(11,11),padding='same', input_shape=(size,size,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))

model.add(Conv2D(filters=256, kernel_size=(5,5), strides=(1,1), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))

model.add(Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), padding='same', activation='relu'))

model.add(Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), padding='same', activation='relu'))

model.add(Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), padding='same', activation='relu'))

model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
model.add(Flatten())
model.add(Dense(4096, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(2048, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(2,activation='softmax')) #2 classes


# In[20]:


model.summary()


# Train

# In[21]:


from keras.callbacks import EarlyStopping,ModelCheckpoint
model.compile(loss='categorical_crossentropy',optimizer='Adam',metrics=['accuracy'])
earlyStopping = EarlyStopping(monitor='val_loss', patience=10)
modelCheckpoint = ModelCheckpoint('model.h5', monitor='val_loss', verbose=1, save_best_only=True, mode='min')


history = model.fit(X_train_norm, y_train_onehot, validation_split=0.0, validation_data=(X_v_norm, y_v_onehot),
                    epochs=50, batch_size=32, verbose=1,
                   callbacks=[earlyStopping, modelCheckpoint])


# In[23]:


# model.save('model.h5')


# In[26]:

from keras.models import load_model
model = load_model('model.h5')


# In[27]:


model.evaluate(X_t_norm, y_t_onehot)


# In[28]:


from sklearn.metrics import classification_report

# y_pred = model.predict_classes(X_test_norm)
y_pred = np.argmax(model.predict(X_t_norm), axis=-1)
print(classification_report(tclasses, y_pred))


# In[29]:


def plot_train_history(history, train_metrics, val_metrics):
    plt.plot(history.history.get(train_metrics),'-o')
    plt.plot(history.history.get(val_metrics),'-o')
    plt.ylabel(train_metrics)
    plt.xlabel('Epochs')
    plt.legend(['train', 'validation'])
    #plt.show()


# In[30]:


plt.figure(figsize=(12,4))
plt.subplot(1,2,1)
plot_train_history(history, 'loss','val_loss')
plt.subplot(1,2,2)
plot_train_history(history, 'accuracy','val_accuracy')
plt.show()
# In[ ]:




