import os
import cv2
import numpy as np

# Iterate through each image
directory = 'MRIheart'
for filename in os.listdir(directory):
    if filename.endswith(".png"):
        img = cv2.imread(directory + '/' + filename)
        final_img = img.copy()
        
        # Calculate the inner ventricle
        roi = img[250:450, 180:360]
        inner_grey = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        ret, inner_thresh = cv2.threshold(inner_grey, 50, 255, cv2.THRESH_BINARY)

        kernel = np.ones((5,5),np.uint8)

        inner_dilation = cv2.dilate(inner_thresh,kernel,iterations=2)
        inner_closing = cv2.morphologyEx(inner_dilation, cv2.MORPH_CLOSE, kernel)
        inner_edge = cv2.Canny(inner_closing, 100, 200)

        inner_contours, hierarchy = cv2.findContours(inner_edge, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        for contour in inner_contours:
            if cv2.contourArea(contour) > 7400:
                cv2.drawContours(final_img[250:450, 180:360], contour, -1, (30, 0, 255), 1)
                inner_ventricle = cv2.contourArea(contour)

        # Calculate the outer ventricle
        roi = cv2.circle(roi, (95, 110), 73, (0,0,0), thickness=-1)
        roi = cv2.circle(roi, (96, 107), 84, (255,255,255), thickness=3)
        
        grey = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(roi, 39, 255, cv2.THRESH_BINARY)

        kernel = np.ones((6,6),np.uint8)

        closing = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

        edge = cv2.Canny(closing, 100, 200) 

        contours, hierarchy = cv2.findContours(edge, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        for contour in contours:
            if cv2.contourArea(contour) > 1000:
                cv2.drawContours(final_img[250:450, 180:360], contour, -1, (30, 0, 255), 1)
                outer_ventricle = cv2.contourArea(contour)  

        cv2.imshow('Final', final_img[250:450, 180:360])
        cv2.waitKey(0)

        
