import argparse
import imutils
import mtcnn
import cv2
import pdb
import os
from tensorflow.keras.models import load_model
import json
import numpy as np
import tensorflow as tf
from face_detection import detect_face_draw, detect_face_on_frames
#tf.config.threading.set_inter_op_parallelism_threads(3)

#python face_image.py --model "face_detector_sinhvien.h5" --classes "class_sinhvien.json" tra_4.jpg
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help='Path of file .h5', default=0)
    parser.add_argument('--classes', type=str, help='Path of file .json')
    parser.add_argument('data', type=str, help='Link data test')
    args = parser.parse_args()


    current_dir = os.path.dirname(os.path.realpath(__file__))
    print(current_dir)
    

    if os.path.isfile(current_dir + "/Classes/{}".format(args.classes)):
        with open(current_dir + "/Classes/{}".format(args.classes), 'r') as file:
            classes = json.load(file)
    else:
        print("There are no classes. Please insert classes.json file")
        exit()
        
    if os.path.isfile(current_dir + "/model/{}".format(args.model)):
        model = load_model(current_dir + "/model/{}".format(args.model))      
    else:
        print("There are no {} file in folder model.".format(args.mode))
        exit()   
        
    # khoi tạo mạng MTCNN
    face_detector = mtcnn.MTCNN()
    
    image = cv2.imread(current_dir+"/datatest/"+args.data)
    max_thesold = 0.5
    image = detect_face_draw(face_detector, image, model, classes, max_thesold = max_thesold)
    image = imutils.resize(image, width=500)
    cv2.imshow("face",image)
    cv2.waitKey(20)
    
    while (True):
        print("Press Q to exit() or enter you image file to recognize: ")
        string = input()

        if string.lower() == "q":
            break
            
        if  not os.path.isfile(current_dir+"/datatest/"+string):
            continue    
            
        image = cv2.imread(current_dir+"/datatest/"+string)
    
        image = detect_face_draw(face_detector, image, model, classes)
        image = imutils.resize(image, width=500)
        
        cv2.imshow("face",image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


    cv2.destroyAllWindows()
