# Vgg16FaceRecognize
Using VGG16 to Recognize Faces. Research with Thay Viet (thviet79), Tra Oliver (Traoliver)

![image](https://user-images.githubusercontent.com/52736465/118868023-ea353100-b90d-11eb-80b3-fa5829ecdecb.png)

Clone mtcnn project from https://github.com/ipazc/mtcnn/tree/master/mtcnn to detect human faces.pro

On this project, we trained model to recognize Face on 3 dataset sample AT&T, Yale and one outer dataset ( students - 26 person - 11-20 images/person)   

Preview data
Yale:
![image](https://user-images.githubusercontent.com/52736465/118865720-7134da00-b90b-11eb-88f2-a9e726ca46ec.png)
AT&T:
![image](https://user-images.githubusercontent.com/52736465/118864090-d687cb80-b909-11eb-82cc-6027a0b4fde5.png)
Student:
  - This dataset was crawled from facebook (for research only) and cropped to satisfy conditions 1 image had only 1 person . 
    After that we use MTCNN to extract faces from there dataset. The difficult of there dataset was it contains multi part of face.
    
![image](https://user-images.githubusercontent.com/52736465/118864676-69286a80-b90a-11eb-9c0b-c63817fbf2e7.png)
![image](https://user-images.githubusercontent.com/52736465/118866090-dbe61580-b90b-11eb-9e57-b84b20dc96b9.png)
![image](https://user-images.githubusercontent.com/52736465/118866170-edc7b880-b90b-11eb-91fc-cb9c0648fcf7.png)
![image](https://user-images.githubusercontent.com/52736465/118866295-0a63f080-b90c-11eb-9835-eaa5559c24d1.png)
![image](https://user-images.githubusercontent.com/52736465/118866354-1a7bd000-b90c-11eb-900b-2c558906c8ab.png)



Here is the result after training:
AT&T:
![image](https://user-images.githubusercontent.com/52736465/118865525-392d9700-b90b-11eb-82de-2ab120f0dbac.png)
Yale:
![image](https://user-images.githubusercontent.com/52736465/118865634-58c4bf80-b90b-11eb-93e3-5d986ae3505d.png)


Student:
Id - 17103100397
![image](https://user-images.githubusercontent.com/52736465/118862789-501eba00-b908-11eb-9a35-0306a9c49948.png)
Id - 17103100414
![image](https://user-images.githubusercontent.com/52736465/118863497-2ca83f00-b909-11eb-9652-3d67b45ed358.png)



