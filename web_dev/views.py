from django.shortcuts import render
import os
import cv2
from fer import FER
import tensorflow as tf
def model():
        emotion_detector = FER()

        face_haar_cascade = cv2.CascadeClassifier(cv2.data.haarcascades +'haarcascade_frontalface_default.xml')    

        cap=cv2.VideoCapture(0)  #select the default video capture

        #If the camera was not opened sucessfully
        if not cap.isOpened():  
            print("Cannot open camera")
            exit()

        is_happy = False

        #Continously read the frames 
        while cap.isOpened():
            #read frame by frame and get return whether there is a stream or not
            ret, frame=cap.read()
            
            #If no frames recieved, then break from the loop
            if not ret:
                print("Can't receive frame (stream end?). Exiting ...")
                break
                
            #Change the frame to greyscale  
            gray_image= cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            #We pass the image, scaleFactor and minneighbour
            faces_detected = face_haar_cascade.detectMultiScale(gray_image,1.32,5)
            
            #Draw Rectangles around the faces detected
            for (x,y,w,h) in faces_detected:
                cv2.rectangle(frame,(x,y), (x+w,y+h), (255,0,0), thickness=3)
                #Get the prediction of the model
                predictions = emotion_detector.detect_emotions(frame)
                # print(predictions[0]['emotions'])
                a_dictionary = predictions[0]['emotions']
                emotion_prediction = max(a_dictionary, key = a_dictionary.get)
                
                print(emotion_prediction)
                
                #Write on the frame the emotion detected
                cv2.putText(frame,emotion_prediction,(int(x), int(y)),cv2.FONT_HERSHEY_SIMPLEX,2,(0,255,0),3)

                if emotion_prediction == 'happy':
                    ret, frame2=cap.read()
                    # cv2.imshow("Capturing", frame2)
                    # cv2.imwrite(filename='saved_img.png', img=frame2)
                    from web_dev import mask
                    # print(frame2)
                    mask.Mask(frame2)
                    print("Image Saved")
                    is_happy = True



            resize_image = cv2.resize(frame, (1000, 700))
            cv2.imshow('Emotion',resize_image) 
            if cv2.waitKey(10) == ord('b') or is_happy:
                    break

        cap.release()
        cv2.destroyAllWindows()
        from web_dev import cartoonify
        cartoonify.my_func('./static/remove_background.jpg')
def home(request):    
    return render(request, 'index.html')
def result(request):
    model()
    return render(request,'result.html')