import matplotlib.pyplot as plt
import tensorflow as tf
import pdb
from tensorflow.keras.layers import Conv2D,Convolution2D, MaxPooling2D, Flatten, Dense,Activation,BatchNormalization,Dropout, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import Model
from  tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import EarlyStopping
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import RMSprop
import triplet_loss_func

if __name__ == '__main__':
    initial_model = tf.keras.applications.VGG16(
        include_top=False, weights='imagenet', input_tensor=None,
        input_shape=(224,224,3), pooling=None, classes=40,
        
    )

    for layer in initial_model.layers:
        layer.trainable = False
    last = initial_model.output

    last = Flatten()(last)
    #last = Dense(4096, activation='relu')(last)
    #last = Dropout(0.1)(last)
    #last = Dense(4096, activation='relu')(last)
    last = Dropout(0.25)(last)
    
    #last = Dense(256, activation='relu')(last)
    #last = Dense(158, activation='relu')(last)
    last = Dense(15, activation='softmax')(last)

    model = Model(inputs = initial_model.input, outputs =last )

    print(model.summary())
    current_dir = os.path.dirname(os.path.realpath(__file__))

    train_generator = ImageDataGenerator(
        #rescale = 1./255,
        #fill_mode = 'nearest',
        fill_mode = "constant",
        validation_split=0.4,
    )

    batch_size = 2
    
    train_gen = train_generator.flow_from_directory(
        current_dir+"/dataset/train",
        #color_mode="grayscale",
        target_size = (224,224),
        batch_size = batch_size, 
        #class_mode='input',
        subset='training'
    )

    val_gen = train_generator.flow_from_directory(
        current_dir+"/dataset/train",
        #color_mode="grayscale",
        target_size = (224,224),
        batch_size = batch_size, 
        #class_mode='input',
        subset='validation'
    )
   
    check_point = ModelCheckpoint("face_detector_yale.h5"
                                    ,monitor="val_loss"
                                    ,mode="min"
                                    ,save_best_only = True,
                                    verbose=1)
    earlystop = EarlyStopping(
    monitor='val_loss',
    min_delta=0,
    patience=3, 
    verbose=1,
    restore_best_weights=True
    )

    callbacks = [earlystop, check_point]
    #callbacks = [check_point]
    model.compile( loss = 'categorical_crossentropy',
                  optimizer = "nadam",
                  metrics = ['accuracy']
    )

    hist = model.fit(
        train_gen,
        epochs = 20,
        callbacks = callbacks,
        validation_data = val_gen
    )

    #model.save("model/face_detector_yale.h5")
    
    plt.plot(hist.history["accuracy"]*100)
    plt.plot(hist.history['val_accuracy']*100)
    plt.plot(hist.history['loss'])
    plt.plot(hist.history['val_loss'])
    plt.title("model accuracy")
    plt.ylabel("Accuracy")
    plt.xlabel("Epoch")
    plt.legend(["Accuracy","Validation Accuracy","loss","Validation Loss"])
    plt.show()