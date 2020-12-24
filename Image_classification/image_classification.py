from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dense, Conv2D, Dropout, MaxPooling2D, Flatten, BatchNormalization, Activation
import PIL.Image as pilimg
from keras import regularizers
from tensorflow.keras.applications import VGG16, VGG19, Xception, ResNet50, ResNet50V2, ResNet101, ResNet101V2, ResNet152, ResNet152V2, InceptionResNetV2
from cv2 import cv2

np.random.seed(3)

train_datagen = ImageDataGenerator(rescale = 1/255., horizontal_flip = True, vertical_flip=True,
                                   width_shift_range = 0.1, height_shift_range = 0.1, 
                                   zoom_range = 0.1,rotation_range = 30, fill_mode = 'nearest')

test_datagen = ImageDataGenerator(rescale = 1/255.)
pred_datagen = ImageDataGenerator(rescale = 1/255.)

# # 1. 데이터
xy_train = train_datagen.flow_from_directory('D:/ImgDetection/KHL/Train', target_size = (224, 224), batch_size = 88, class_mode = 'binary') 
                                                    
xy_test = test_datagen.flow_from_directory('D:/ImgDetection/KHL/Test', target_size = (224, 224), batch_size = 18, class_mode = 'binary')

predict = pred_datagen.flow_from_directory('D:/ImgDetection/KHL/Predict', target_size = (224, 224), batch_size = 42, class_mode = None)

# next(xy_train)

np.save('D:/ImgDetection/KHL/npy/train_x.npy', arr = xy_train[0][0])
np.save('D:/ImgDetection/KHL/npy/train_y.npy', arr = xy_train[0][1])
np.save('D:/ImgDetection/KHL/npy/test_x.npy', arr = xy_test[0][0])
np.save('D:/ImgDetection/KHL/npy/test_y.npy', arr = xy_test[0][1])
np.save('D:/ImgDetection/KHL/npy/predict_data.npy', arr = predict[0])

print(xy_train[0][0].shape)
print(xy_train[0][1].shape)
print(xy_test[0][0].shape)
print(xy_test[0][1].shape)
print(predict[0].shape)

# 2. 모델 구성
inception_ResNet = InceptionResNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
inception_ResNet.trainable = False
model = Sequential(inception_ResNet)
# model.add(Conv2D(128, (3, 3), kernel_regularizer=regularizers.l1(0.001), activation='linear', input_shape = (200, 200, 3)))
# model.add(BatchNormalization())
# model.add(Activation('relu'))
# model.add(MaxPooling2D(pool_size=(3,3)))

# model.add(Conv2D(64, (3, 3), kernel_regularizer=regularizers.l1(0.001), activation='linear'))
# model.add(BatchNormalization())
# model.add(Activation('relu'))
# model.add(MaxPooling2D(pool_size=(3,3)))

# model.add(Conv2D(128, (3, 3), kernel_regularizer=regularizers.l1(0.001), activation='linear'))
# model.add(BatchNormalization())
# model.add(Activation('relu'))
# model.add(MaxPooling2D(pool_size=(3,3)))

model.add(Flatten())   

model.add(Dense(100, kernel_initializer='he_normal'))     
model.add(BatchNormalization())
model.add(Activation('relu')) 

model.add(Dense(30, kernel_initializer='he_normal'))                                    
model.add(BatchNormalization())
model.add(Activation('relu'))

model.add(Dense(8, kernel_initializer='he_normal'))                                    
model.add(BatchNormalization())
model.add(Activation('relu'))

model.add(Dense(1, activation = 'sigmoid'))

model.summary()

# 3. 컴파일, 훈련
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint 
# model = load_model('D:/ImgDetection/KHL/Checkpoint/CheckPoint-88- 0.008479.hdf5')   
modelpath = "D:/ImgDetection/KHL/Checkpoint/CheckPoint-{epoch:02d}-{val_loss: 4f}.hdf5"  

early_stopping = EarlyStopping(monitor = 'val_loss', patience = 30)
cp = ModelCheckpoint(filepath = modelpath, monitor = 'val_loss', save_best_only = True, mode = 'auto')

model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
hist = model.fit_generator(xy_train, steps_per_epoch = 50, epochs = 100, validation_data = xy_test, validation_steps = 10, callbacks=[early_stopping, cp])

# 4. 평가, 예측
loss, accuracy = model.evaluate(xy_test)
print("loss : ", loss)
print("accuracy : ", accuracy)

y_pred = model.predict_generator(predict, steps=1, verbose = 1)
y_pred = np.round(y_pred)
print(y_pred)

# Matplotlib을 활용한 예측값과 실제값 시각화
fig = plt.figure()
rows = 7
cols = 10

a = ['Nam', 'Suzy']

def printIndex(array, i):
    if array[i][0] == 0:
        return a[0]
    elif array[i][0] == 1:
        return a[1]

for i in range(len(predict[0])):
    ax = fig.add_subplot(rows, cols, i+1)
    ax.imshow(predict[0][i])
    label = printIndex(y_pred, i)
    ax.set_xlabel(label)
    ax.set_xticks([]), ax.set_yticks([])
plt.show()

