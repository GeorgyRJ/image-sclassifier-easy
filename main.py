import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import os 
import random 
import warnings
from tensorflow.keras.preprocessing.image import load_img 

warnings.filterwarnings("ignore")

## create a datafream for input and output

input_part = []
label = []

for class_name in os.listdir("petImages"):
    for path in os.listdir("petImages/"+class_name): 
        if class_name == 'cat':
            label.append(0)
        else :
            label.append(1)
        input_part.append(os.path.join("petImages",class_name,path))
print(input_part[0],label[0])

## show the dataframe

df = pd.DataFrame()
df['images'] = input_part
df['label'] = label
df = df.sample(frac=1).reset_index(drop=True)
df.head()

for i in df['images']:
    if '.jpg' not in i :
        print (i)

import PIL
import PIL.Image
I = []
for image in df ['images']:
    try :
        img = PIL.Image.open(image)
    except :
        I.append(image)

df = df[df['images']!='petImages\Cat\Thumbs.db']
df = df[df['images']!='petImages\Dog\Thumbs.db']
df = df[df['images']!='petImages\Dog\11702.jpg']
df = df[df['images']!='petImages\Cat\666.jpg']
len(df)
## explorstory data anylysis
#dog
#display grid of images
plt.figure(figsize=(25,25))
temp = df[df['label']==1]['images']
start = random.randint(0,len(temp))
files = temp[start:start+25]


for index,file in enumerate(files):
    plt.subplot(5,5,index+1)
    img = load_img(file)
    img = np.array(img)
    plt .imshow(img)
    plt.title('Dogs')
    plt.axis('off')

#cat 

plt.figure(figsize=(25,25))
temp = df[df['label']==0]['images']
start = random.randint(0,len(temp))
files = temp[start:start+25]

for index,file in enumerate(files):
    plt.subplot(5,5,index+1)
    img = load_img(file)
    img = np.array(img)
    plt .imshow(img)
    plt.title('Cats')
    plt.axis('off')

import seaborn as sns 
sns.countplot(df['label'])

print(df['label'].value_counts())

##Create DataGenerator for the Images 

df['label']=df['label'].astype('str')
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_generator = ImageDataGenerator(
    rescale=1./255,  # normalization of images
    rotation_range=40,  # augmentation of images to avoid overfitting
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest' 
    )

val_generator = ImageDataGenerator(rescale=1./255)

train_iterator = train_generator.flow_from_dataframe(
    train,
    x_col='images',
    y_col='label',
    target_size = (128,128),
    batch_size = 512   , ## can change size upto your computer (64/128/512 )
    class_mode = 'binary'
    )

val_iterator = val_generator.flow_from_dataframe(
    test,
    x_col='images',
    y_col='label',
    target_size = (128,128),
    batch_size = 512   , ## can change size upto your computer (64/128/512 )
    class_mode = 'binary'
    )



from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

## make a model 

model = Sequential([
    Conv2D(16,(3,3),activation='relu',input_shape=(128,128,3)), ## 1st layer
    MaxPooling2D((2,2)),
    Conv2D(32,(3,3),activation='relu'),
    MaxPooling2D((2,2)),
    Conv2D(64,(3,3),activation='relu'),
    MaxPooling2D((2,2)),
    Flatten(),
    Dense(512,activation='relu'),
    Dense(1,activation='sigmoid')
])

## compile the model 

model.compile(optimizer = 'adam',loss='binary_cross_entropy',metrics=['accuracy'])
model.summary()

history = model.fit(
    train_iterator,
    epochs = 3 ,
    validation_data = val_iterator
)