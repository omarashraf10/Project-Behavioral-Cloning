import csv
import cv2
import numpy as np
from scipy import ndimage

images=[]
measurments=[]
lines=[]
with open ('/opt/my-collected-data/data_driving_log.csv') as csvfile:
    reader =csv.reader(csvfile)
    
    for line in reader:
        lines.append(line)


for line in lines:
    source_path=line[0]  
    filename=source_path.split('/')[-1]
    current_path = '/opt/my-collected-data/imgs/' +filename
    image = ndimage.imread(current_path)
    images.append(image)
    measurment=float(line[3])
    measurments.append(measurment)

lines=[]
with open ('/opt/my-collected-data/tricky_driving_log.csv') as csvfile:
    reader =csv.reader(csvfile)
    
    for line in reader:
        lines.append(line)


for line in lines:
    source_path=line[0]  
    filename=source_path.split('/')[-1]
    current_path = '/opt/my-collected-data/tricky_imgs/' +filename
    image = ndimage.imread(current_path)
    images.append(image)
    measurment=float(line[3])
    measurments.append(measurment)
 
lines=[]
with open ('/opt/my-collected-data/tricky2_driving_log.csv') as csvfile:
    reader =csv.reader(csvfile)
    
    for line in reader:
        lines.append(line)


for line in lines:
    source_path=line[0]  
    filename=source_path.split('/')[-1]
    current_path = '/opt/my-collected-data/tricky2_imgs/' +filename
    image = ndimage.imread(current_path)
    images.append(image)
    measurment=float(line[3])
    measurments.append(measurment)


    
augmented_images, augmented_measurments=[],[]
for image,measurment in zip(images,measurments):
    augmented_images.append(image)
    augmented_measurments.append(measurment)
    augmented_images.append(cv2.flip(image,1))
    augmented_measurments.append(measurment*-1.0)
    
    
x_train=np.array(augmented_images)
y_train=np.array(augmented_measurments)

from keras.models import Sequential
from keras.layers import Lambda
from keras.layers.core import Dense, Activation, Flatten, Dropout
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.layers import Cropping2D

model = Sequential()
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((70,25),(0,0))))
model.add(Conv2D(24,5,5,subsample=(2,2), activation="relu"))
model.add(Dropout(0.5))
model.add(Conv2D(36,5,5,subsample=(2,2), activation="relu"))
model.add(Dropout(0.5))
model.add(Conv2D(48,5,5,subsample=(2,2), activation="relu"))
model.add(Conv2D(64,3,3,activation="relu"))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(100))
model.add(Dropout(0.5))
#model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit(x_train,y_train,validation_split=0.2,shuffle=True,nb_epoch=3)
model.save('model.h5')