import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D
from keras.layers import Dense, Activation, Dropout, Flatten
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import array_to_img
import numpy as np
import matplotlib.pyplot as plt
import cv2

def Transfer_learning():
    #------------------------------
    #cpu - gpu configuration
    config = tf.ConfigProto( device_count = {'GPU': 0 , 'CPU': 56} ) #max: 1 gpu, 56 cpu
    sess = tf.Session(config=config) 
    keras.backend.set_session(sess)
    #------------------------------
    #variables
    num_classes = 7 #angry, disgust, fear, happy, sad, surprise, neutral
    #------------------------------
    #construct CNN structure
    model = Sequential()

    #1st convolution layer
    model.add(Conv2D(64, (5, 5), activation='relu', input_shape=(48,48,1)))
    model.add(MaxPooling2D(pool_size=(5,5), strides=(2, 2)))

    #2nd convolution layer
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(AveragePooling2D(pool_size=(3,3), strides=(2, 2)))

    #3rd convolution layer
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(AveragePooling2D(pool_size=(3,3), strides=(2, 2)))

    model.add(Flatten())

    #fully connected neural networks
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.2))

    model.add(Dense(num_classes, activation='softmax'))
    #------------------------------

    model.compile(loss='categorical_crossentropy'
        , optimizer=keras.optimizers.Adam()
        , metrics=['accuracy']
    )

    #------------------------------

    model.load_weights('Data/facial_expression_model_weights.h5') #load weights
    return model


#function for drawing bar chart for emotion preditions
def emotion_analysis(emotions):
    objects = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')
    y_pos = np.arange(len(objects))

def Sentiment_Analysis(image, model):

    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    x = cv2.resize(gray_image, (48, 48))
    x = np.expand_dims(x, axis = -1)
    x = np.expand_dims(x, axis = 0)
    x = np.divide(x, 255)
    custom = model.predict(x)
    emotion_analysis(custom[0])

    objects = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')
    Sentiment = objects[np.argmax(custom)]
    Percentage = max(custom[0])*100


    return Sentiment, Percentage
    
    
    
