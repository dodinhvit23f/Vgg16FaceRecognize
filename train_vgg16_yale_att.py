import os
import tensorflow as tf
from tensorflow.keras.layers import Conv2D,Convolution2D, MaxPooling2D, Flatten, Dense,Activation,BatchNormalization,Dropout, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import Model
from  tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import EarlyStopping
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import RMSprop
print(tf.__version__)
import matplotlib.pyplot as plt
import json


initial_model = tf.keras.applications.VGG16(
    include_top=False, weights='imagenet', input_tensor=None,
    input_shape=(224, 224, 3), pooling=None, classes=36,
)

for layer in initial_model.layers:
    layer.trainable = False
last = initial_model.output

last = Flatten()(last)
# last = Dense(4096, activation='relu')(last)
# last = Dropout(0.1)(last)
# last = Dense(4096, activation='relu')(last)
last = Dropout(0.3)(last)

# last = Dense(256, activation='relu')(last)
# last = Dense(158, activation='relu')(last)
last = Dense(36, activation='softmax')(last)

model = Model(inputs=initial_model.input, outputs=last)

print(model.summary())
# current_dir = os.path.dirname(os.path.realpath(__file__))

train_generator = ImageDataGenerator(
    # rescale = 1./255,
    # fill_mode = 'nearest',
    fill_mode="constant",
    validation_split=0.2,
    width_shift_range=0.1,
    height_shift_range=0.1,
)
"""
val_generator = ImageDataGenerator(
    #rescale = 1./255,
    #fill_mode = 'nearest',
    fill_mode = "constant",
    validation_split=0.3,
)
"""

batch_size = 16

train_gen = train_generator.flow_from_directory(
    "Dataset/FaceData/processed",
    # color_mode="grayscale",
    target_size=(224, 224),
    batch_size=batch_size,
    # class_mode='input',
    subset='training'
)

val_gen = train_generator.flow_from_directory("Dataset/FaceData/processed",
                                              # color_mode="grayscale",
                                              target_size=(224, 224),
                                              batch_size=batch_size,
                                              # class_mode='input',
                                              subset='validation'
                                              )

dict_ = val_gen.class_indices
class_dict = dict()

for key in dict_:
    class_dict[dict_[key]] = key

with open("class.json", "w") as f:
    json.dump(class_dict, f)
check_point = ModelCheckpoint("face_detector.h5"
                              , monitor="val_loss"
                              , mode="min"
                              , save_best_only=True,
                              verbose=1)
earlystop = EarlyStopping(
    monitor='val_loss',
    min_delta=0,
    patience=5,
    verbose=1,
    restore_best_weights=True
)

callbacks = [earlystop, check_point]
# callbacks = [check_point]
model.compile(loss='categorical_crossentropy',
              optimizer="nadam",
              metrics=['accuracy']
              )

hist = model.fit(
    train_gen,
    epochs=20,
    callbacks=callbacks,
    validation_data=val_gen
)

model.save("face_detector_uneti.h5")

plt.plot(hist.history["accuracy"])
plt.plot(hist.history['val_accuracy'])
plt.title("model accuracy")
plt.ylabel("Accuracy")
plt.xlabel("Epoch")
plt.legend(["Accuracy", "Validation Accuracy"])
plt.show()

plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title("model loss")
plt.ylabel("Loss")
plt.xlabel("Epoch")
plt.legend(["Loss", "Validation Loss"])
plt.show()