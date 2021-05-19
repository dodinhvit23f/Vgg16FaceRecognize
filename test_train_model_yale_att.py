<<<<<<< HEAD
import os
import cv2
import tensorflow as tf
import numpy as np
import pdb
#import train
from tensorflow.keras.applications.resnet50 import decode_predictions
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
if __name__ == '__main__':
    model = load_model('DanhSachGiangVienCNTT/Models/face_detector_uneti.h5')

    current_dir = os.path.dirname(os.path.realpath(__file__))
        
    test_generator = ImageDataGenerator()
    test_folder = current_dir+"/DanhSachGiangVienCNTT/Dataset/FaceData/raw"
    list_dir = os.listdir(test_folder)
    
    train_generator = ImageDataGenerator(
        #rescale = 1./255,
        #fill_mode = 'nearest',
        validation_split=0.6,
    )
    
  
    
    val_gen = train_generator.flow_from_directory(
        current_dir+"/DanhSachGiangVienCNTT/Dataset/FaceData/processed",
        #color_mode="grayscale",
        target_size = (224,224),
        batch_size = 1, 
        class_mode='input',
        subset='validation'
    )
    
    dict_ = val_gen.class_indices
    print( dict_)
    
    for file in list_dir:
        if ( file.find(".") != -1):
            if ( file.find(".py") != -1):
                continue
                
            image = cv2.imread(test_folder+"/"+file) 
            image = cv2.resize(image, (224,224))
    
            pre = model.predict(np.array([image]))
            res = np.argmax(pre, axis=1)
            for key in dict_:
                if (dict_[key] == res):
                    print("{}\t {} \t {}".format(key, pre[0][res[0]] * 100, file))
                    break
            cv2.waitKey(0)
    
    #print('Predicted:', decode_predictions(perdict, top=3)[0])
=======
import os
import cv2
import tensorflow as tf
import numpy as np
import pdb
#import train
from tensorflow.keras.applications.resnet50 import decode_predictions
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
if __name__ == '__main__':
    model = load_model('DanhSachGiangVienCNTT/Models/face_detector_uneti.h5')

    current_dir = os.path.dirname(os.path.realpath(__file__))
        
    test_generator = ImageDataGenerator()
    test_folder = current_dir+"/DanhSachGiangVienCNTT/Dataset/FaceData/raw"
    list_dir = os.listdir(test_folder)
    
    train_generator = ImageDataGenerator(
        #rescale = 1./255,
        #fill_mode = 'nearest',
        validation_split=0.6,
    )
    
  
    
    val_gen = train_generator.flow_from_directory(
        current_dir+"/DanhSachGiangVienCNTT/Dataset/FaceData/processed",
        #color_mode="grayscale",
        target_size = (224,224),
        batch_size = 1, 
        class_mode='input',
        subset='validation'
    )
    
    dict_ = val_gen.class_indices
    print( dict_)
    
    for file in list_dir:
        if ( file.find(".") != -1):
            if ( file.find(".py") != -1):
                continue
                
            image = cv2.imread(test_folder+"/"+file) 
            image = cv2.resize(image, (224,224))
    
            pre = model.predict(np.array([image]))
            res = np.argmax(pre, axis=1)
            for key in dict_:
                if (dict_[key] == res):
                    print("{}\t {} \t {}".format(key, pre[0][res[0]] * 100, file))
                    break
            cv2.waitKey(0)
    
    #print('Predicted:', decode_predictions(perdict, top=3)[0])
>>>>>>> 42aa578ac428127d872dc85a6300b8577fdbf61c
    