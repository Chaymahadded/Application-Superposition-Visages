from pprint import pprint
import dlib
import cv2
import numpy as np
import csv
from PIL import Image
from matplotlib import pyplot as plt

img = cv2.imread(('9.png'))
 
result = img.astype(np.float32)
result = result / 255.0
print(result)


detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("C:\\Users\HP\Downloads\shape_predictor_68_face_landmarks.dat")

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

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
                        cv2.circle(img ,(x , y), 3 , (255,0,0) , -1)
                        cv2.putText(img,str(k),(int (x ) + 5, int (y) + 5 ),fontFace=cv2.FONT_HERSHEY_SIMPLEX,fontScale=0.5,color=(0, 0, 0),)
                        cv2.imshow("frame", img)
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
                with open('sunglasses_5.csv') as csv_file:
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
                glasses_img = cv2.imread('sunglasses_5.jpg')
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

                #print(transformed_glasses)
                cv2.imshow('transdormed',transformed_glasses)
                # mask overlay

                alpha_glasses = transformed_glasses[:,:,2]
                alpha_image = 1.0 - alpha_glasses

                for c in range(0, 2):
                        result[:, :, c] = (
                            alpha_glasses * transformed_glasses[:, :, c]
                            + alpha_image * result[:, :, c]
                        )
cv2.imshow("frame",result)
plt.figure()
plt.imshow(result) 
plt.show()
