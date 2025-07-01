import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, Flatten, Dense, Dropout
from tensorflow.keras.metrics import Precision, Recall


def create_cnn(pool_type='max', conv_activation='sigmoid', dropout_rate=0.10):
    model = Sequential()
    
    model.add(Conv2D(16, kernel_size=(5, 5), activation='relu', input_shape=(32, 32, 3))) 
        
    model.add(Conv2D(32, kernel_size=(5, 5), activation=conv_activation))  
    if pool_type == 'max':
        model.add(MaxPooling2D(pool_size=(2, 2)))
    elif pool_type == 'average':
        model.add(AveragePooling2D(pool_size=(2, 2)))
    if dropout_rate != 0:
        model.add(Dropout(rate=dropout_rate))     
    
    model.add(Conv2D(64, kernel_size=(3, 3), activation=conv_activation)) 
    if pool_type == 'max':
        model.add(MaxPooling2D(pool_size=(2, 2)))
    elif pool_type == 'average':
        model.add(AveragePooling2D(pool_size=(2, 2)))
    if dropout_rate != 0:
        model.add(Dropout(rate=dropout_rate))     
      
    model.add(Flatten())         
    model.add(Dense(64, activation='sigmoid')) # 64
    if dropout_rate != 0:
        model.add(Dropout(rate=dropout_rate)) 
        
    model.add(Dense(6, activation='softmax'))
    
    model.compile( 
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy'],
        )    
    return model
