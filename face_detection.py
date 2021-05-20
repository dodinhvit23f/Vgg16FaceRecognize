import mtcnn
import cv2
import pdb
import os
from tensorflow.keras.models import load_model
import json
import numpy as np
import tensorflow as tf
import argparse
#tf.config.threading.set_inter_op_parallelism_threads(3)

def detect_face_draw (face_detector, image, model= None, classes = None, max_thesold = 0.8 ):

    conf_trust = 0.98
    
    
    results = face_detector.detect_faces(image)

    for result in results:
    
        confidence = result['confidence']
        
        if confidence < conf_trust:
               continue
               
        x , y , width, height = result['box']

        x_end, y_end = x + width, y + height

        key_points = result['keypoints']
              
        #cv2.circle(image, key_points['left_eye'], radius=2, color=(0, 255, 0), thickness=-1)
        #cv2.circle(image, key_points['right_eye'], radius=2, color=(0, 255, 0), thickness=-1)
        #cv2.circle(image, key_points['mouth_left'], radius=2, color=(0, 255, 0), thickness=-1)
        #cv2.circle(image, key_points['mouth_right'], radius=2, color=(0, 255, 0), thickness=-1)
        #cv2.circle(image, key_points['nose'], radius=2, color=(0, 255, 0), thickness=-1)
        
        
        
        if(model != None):
            if(classes == None):
                raise("when mode not None class must decalare")
                
            face = image[ y:  y_end , x : x_end]
            line = 20
            color = (0,0,255)
            
            face = cv2.resize(face, (224,224))
            
            #cv2.imshow("face",face)
            #cv2.waitKey(1)
            
            pre = model.predict(np.array([face]))
            res = np.argmax(pre, axis=1)
            
            if( pre[0][res[0]] < max_thesold):
                label = "unknow"
            else:    
                label = classes[str(res[0])]
                
                                   
                image = cv2.putText(image, 
                    "{}".format(pre[0][res[0]]) ,
                    (x, y_end + line*2),
                    color= color,
                    fontScale = 0.8,
                    thickness = 1,
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX )
                    
            image = cv2.putText(image, 
                    label ,
                    (x, y_end + line ),
                    color= color,
                    fontScale = 0.8,
                    thickness = 1,
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX )
            print("{}-{}-{}".format(label, pre[0][res[0]], classes[str(res[0])] ))        
            label = ""
            res = 0
        #hiển thị khuôn mặt.
        #cv2.imshow("face",face)
        #cv2.waitKey(1)
        cv2.rectangle(image, (x, y), (x_end, y_end), (0, 255, 0), thickness = 2)
    #image = cv2.resize(image,(424,424))
    
    return image

def detect_face_on_frames(face_detector, frame, conf_trust = 0.9):
    
    if( face_detector == None):
        face_detector = mtcnn.MTCNN()
    
    results = face_detector.detect_faces(frame)

    list_face = list()

    for result in results:
        x , y , width, height = result['box']

        x_end, y_end = x + width, y + height

        confidence = result['confidence']

        if confidence < conf_trust:
                continue
        
        key_points = result['keypoints']

        list_face.append(x , y , width, height, key_points['left_eye'], key_points['right_eye'], key_points['mouth_left'], key_points['mouth_right'])

    return list_face


#python face_detection.py --model "face_detector_uneti.h5" -- class "class.json"
"""
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help='Path of file .h5', default=0)
    parser.add_argument('--classes', type=str, help='Path of file .json')
    parser.add_argument('--videos', type=str, help='Link data test')
    args = parser.parse_args()

    current_dir = os.path.dirname(os.path.realpath(__file__))

    model = load_model('model/{}'.format(args.model))
    
    if os.path.exists(current_dir + "/Classes/{}".format(args.classes)):
        with open(current_dir + "/Classes/{}".format(args.classes), 'r') as file:
            classes = json.load(file)

    # khoi tạo mạng MTCNN
    face_detector = mtcnn.MTCNN()
    
    
    cap = cv2.VideoCapture('Videos/{}'.format(args.videos))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
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
"""