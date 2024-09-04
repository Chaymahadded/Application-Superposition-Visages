from pprint import pprint
import dlib
import cv2
import numpy as np
import csv
from PIL import Image



cap = cv2.VideoCapture(0)

detector = dlib.get_frontal_face_detector()

predictor = dlib.shape_predictor("C:\\Users\HP\Downloads\shape_predictor_68_face_landmarks.dat")


while True:
        _, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        result = frame.copy()
        result = result.astype(np.float32) / 255.0

        faces = detector(gray)
        for face in faces:
                x1 = face.left()
                y1 = face.top()
                x2 = face.right()
                y2 = face.bottom()
                
                landmarks = predictor(gray , face)
                for k in range (0 , 68):
                        x = landmarks.part(k).x
                        y = landmarks.part(k).y
                        cv2.circle(frame ,(x , y), 3 , (255,0,0) , -1)
                        cv2.putText(frame,str(k),(int (x ) + 5, int (y) + 5 ),fontFace=cv2.FONT_HERSHEY_SIMPLEX,fontScale=0.5,color=(0, 0, 255),)
                # get chosen landmarks 37-48, 28,1,17 as destination points
                dst_pts=[]
                for k in range(36, 48):
                        x = landmarks.part(k).x
                        y = landmarks.part(k).y

                        dst_pts.append([x,y])
                #create np array from list
                dst_pts= np.array(dst_pts, dtype="float32",)      
                print('*'*100)  
                print(dst_pts)
                print('*'*100)
                #open csv file
                with open('sunglasses_4.csv') as csv_file:
                    csv_reader = csv.reader(csv_file, delimiter=",")
                    src_pts = []
                    for i, row in enumerate(csv_reader):
                        # skip head or empty line if it's there
                        try:
                            src_pts.append(np.array([float(row[1]), float(row[2])]))
                        except ValueError:
                            continue
                src_pts = np.array(src_pts, dtype="float32")
                print('*'*100)
                print(src_pts)
                print('*'*100)

                #import glasses_img
                glasses_img = cv2.imread('sunglasses_4.png')
                glasses_img = glasses_img.astype(np.float32)
                glasses_img = glasses_img / 255.0

                # get the perspective transformation matrix
                M, _ =cv2.findHomography(src_pts, dst_pts)
                
                #transformed glasses image
                transformed_glasses = cv2.warpPerspective(
                        glasses_img,
                        M,
                        (result.shape[1], result.shape[0]),
                        None,
                        cv2.INTER_LINEAR,
                        cv2.BORDER_CONSTANT,
                    )
                # glasses overlay

                alpha_glasses = transformed_glasses[:,:,2]
                alpha_image = 1.0 - alpha_glasses

                for c in range(0, 2):
                        result[:, :, c] = (
                            alpha_glasses * transformed_glasses[:, :, c]
                            + alpha_image * result[:, :, c]
                        )       
        cv2.imshow("Frame" , result)


        key = cv2.waitKey(1)
        if key == 27 :
                break