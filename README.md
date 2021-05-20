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
    After that we use MTCNN to extract faces from there dataset. The difficult of there dataset was it contains multi part and angle of face.
    
![image](https://user-images.githubusercontent.com/52736465/118864676-69286a80-b90a-11eb-9c0b-c63817fbf2e7.png)

![image](https://user-images.githubusercontent.com/52736465/118866090-dbe61580-b90b-11eb-9e57-b84b20dc96b9.png)

![image](https://user-images.githubusercontent.com/52736465/118866170-edc7b880-b90b-11eb-91fc-cb9c0648fcf7.png)

![image](https://user-images.githubusercontent.com/52736465/118866295-0a63f080-b90c-11eb-9835-eaa5559c24d1.png)

![image](https://user-images.githubusercontent.com/52736465/118866354-1a7bd000-b90c-11eb-900b-2c558906c8ab.png)



Here is the result after training:

AT&T(40 person):
  We trained AT&T dataset by divide to 2 part 40% for training and 60% for validation. Train time 7 minute and maximum validation accuracy it 95% after 20 times trained.
  
  ![image](https://user-images.githubusercontent.com/52736465/118943091-821e3380-b97d-11eb-9305-8ec2b3896508.png)

  ![image](https://user-images.githubusercontent.com/52736465/118952346-11c7e000-b986-11eb-99bc-2415d5334e7e.png)

Yale (15 person):
  We trained AT&T dataset by divide to 2 part 40% for training and 60% for validation. Train time 7 minute and maximum validation accuracy it 100% after 20 times trained.
  
  ![image](https://user-images.githubusercontent.com/52736465/118945927-21dcc100-b980-11eb-9ddf-0c76d7934563.png)
  
  ![image](https://user-images.githubusercontent.com/52736465/118952137-e2b16e80-b985-11eb-8c77-e228d612bd39.png)

 AT&t + Yale dataset (55 person):
 We trained AT&T dataset by divide to 2 part 40% for training and 60% for validation. Train time 7 minute and maximum test accuracy it 95.76% after 20 times trained.
  ![image](https://user-images.githubusercontent.com/52736465/118948079-286c3800-b982-11eb-8379-53a4b4ae00c5.png)
  ![image](https://user-images.githubusercontent.com/52736465/118952752-666b5b00-b986-11eb-84a2-4975c4014d6c.png)

Student:
  

Id - 17103100397

![image](https://user-images.githubusercontent.com/52736465/118862789-501eba00-b908-11eb-9a35-0306a9c49948.png)
![image](https://user-images.githubusercontent.com/52736465/118868530-7e06fd00-b90e-11eb-96ee-e339d19b08bf.png)
![image](https://user-images.githubusercontent.com/52736465/118869170-43519480-b90f-11eb-9c7b-0691d61bc83e.png)




Id - 17103100409

![image](https://user-images.githubusercontent.com/52736465/118870430-c0313e00-b910-11eb-800d-9fcde143b510.png)
![image](https://user-images.githubusercontent.com/52736465/118872770-261ec500-b913-11eb-8079-ab978c6f5bee.png)
![image](https://user-images.githubusercontent.com/52736465/118947758-d9be9e00-b981-11eb-9b57-c093b2244797.png)


We also should tested there model with a person non-trained and it respond well
![image](https://user-images.githubusercontent.com/52736465/118869697-ebfff400-b90f-11eb-91f9-519c93ea17bd.png)
![image](https://user-images.githubusercontent.com/52736465/118869827-10f46700-b910-11eb-81bb-eb344d26f10c.png)
![image](https://user-images.githubusercontent.com/52736465/118869887-27022780-b910-11eb-8510-70f0ef96d42a.png)





