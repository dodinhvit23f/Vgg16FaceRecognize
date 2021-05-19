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

#python face_video.py --model "face_detector_sinhvien.h5" --classes "class_sinhvien.json" test1.mp4
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help='Path of file .h5', default=0)
    parser.add_argument('--classes', type=str, help='Path of file .json')
    parser.add_argument('video', type=str, help='Video file')
    args = parser.parse_args()


    current_dir = os.path.dirname(os.path.realpath(__file__))
    
    if os.path.exists(current_dir + "/Classes/{}".format(args.classes)):
        with open(current_dir + "/Classes/{}".format(args.classes), 'r') as file:
            classes = json.load(file)
    else:
        print("There are no classes. Please insert classes.json file")
        exit()
        
    if os.path.exists(current_dir + "/model/{}".format(args.model)):
        model = load_model(current_dir + "/model/{}".format(args.model))      
    else:
        print("There are no {} file in folder model.".format(args.mode))
        exit()   

    # khoi tạo mạng MTCNN
    face_detector = mtcnn.MTCNN()
    
    if not os.path.exists(current_dir + '/Videos/{}'.format(args.video)):
        print("There are no {} file in folder Videos.".format(args.video))
        exit() 
    
    cap = cv2.VideoCapture('Videos/{}'.format(args.video))
    # định dạng lưu lại video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    # Video sẽ lưu lại dưới dạng output.mp4
    stored_video = cv2.VideoWriter('Videos/output.mp4', fourcc, 20.0, (854,480))
    
    while(cap.isOpened()):
        ret, frame = cap.read()
      
        #frame = cv2.resize(frame,(400,400))
        detect_face_draw(face_detector, frame, model, classes)
        cv2.imshow('frame',frame)
        stored_video.write(frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    stored_video.release()
    cv2.destroyAllWindows()