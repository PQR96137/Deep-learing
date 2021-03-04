import os
from PIL import Image
import time
import numpy as np
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.optimizers import SGD, RMSprop, Adam
from keras.layers import Conv2D, MaxPooling2D
import cv2

webcam = cv2.VideoCapture(1)
y=1
def com():
    while y:
        while 1:
            rat, Shooting_images = webcam.read()
            cv2.imshow("1", Shooting_images)
            k = cv2.waitKey(5) & 0xFF
            if k == ord('k'):
                cv2.imwrite(r"C:\Users\user\Desktop\LEGO_image\1\\" + "gear" + str(time.time()) + ".jpg", Shooting_images)
                break
            elif k == ord('b'):
                cv2.destroyAllWindows()

                return 1.5
            elif k == ord('+'):
                return 0.5
        x_test = []
        x_test.clear()
        x_test.append(cv2.resize(Shooting_images, (100, 100)))
        x_test = np.array(x_test)
        print(x_test.shape)
        x_test = x_test.astype('float32')
        x_test /= 255

        model = Sequential()
        model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(100, 100, 3)))
        model.add(Conv2D(32, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))


        model.add(Flatten())
        model.add(Dense(256, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(4, activation='softmax'))

        sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
        model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

        model.load_weights('./weight/weight_loss0.01.hdf5')
        classes = model.predict_classes(x_test)[0]
        '''
        if classes == 0:
            classes=3
        elif classes == 3:
            classes=0
            '''
        target = ['L', 'Circle', 'Square', 'Long strip']
        cv2.putText(Shooting_images, target[classes], (20,90),
                        cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 0),5,cv2.LINE_AA)
        cv2.resize(Shooting_images, (360, 640))
        cv2.imshow("2",Shooting_images)
        print(target[classes])
        print(classes)
        return classes
