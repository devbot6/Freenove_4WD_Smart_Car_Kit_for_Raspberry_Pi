import time
from Motor import *
import RPi.GPIO as GPIO
import yolov5
import cv2
import numpy as np
import os
from picamera2 import Picamera2, Preview

import time
from rpi_ws281x import *

import threading



global cType
from CameraType import CameraType
cType = CameraType()

from Led import *
led=Led()

seen = []




def find_ball(img):
    PWM.setMotorModel(0,0,0,0)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    lower_red = np.array([0, 50, 50])
    upper_red = np.array([20, 255, 255])

    lower_green = np.array([40, 20, 50])
    upper_green = np.array([90, 255, 255])

    lower_blue = np.array([100, 50, 50])
    upper_blue = np.array([130, 255, 255])

    lower_yellow = np.array([20, 20, 50])
    upper_yellow = np.array([50, 255, 130])

    mask_red = cv2.inRange(hsv, lower_red, upper_red)
    mask_green = cv2.inRange(hsv, lower_green, upper_green)
    mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)
    mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)


    contours_red, _ = cv2.findContours(mask_red, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours_green, _ = cv2.findContours(mask_green, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours_blue, _ = cv2.findContours(mask_blue, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours_yellow, _ = cv2.findContours(mask_yellow, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)



    model_name = 'Yolov5_models'
    yolov5_model = 'balls5n.pt'
    model_labels = 'balls5n.txt'

    CWD_PATH = os.getcwd()
    PATH_TO_LABELS = os.path.join(CWD_PATH,model_name,model_labels)
    PATH_TO_YOLOV5_GRAPH = os.path.join(CWD_PATH,model_name,yolov5_model)

    # Import Labels File
    with open(PATH_TO_LABELS, 'r') as f:
        labels = [line.strip() for line in f.readlines()]

    # Initialize Yolov5
    model = yolov5.load(PATH_TO_YOLOV5_GRAPH)

    min_conf_threshold = 0.25
    # set model parameters
    model.conf = 0.25  # NMS confidence threshold
    model.iou = 0.45  # NMS IoU threshold
    model.agnostic = False  # NMS class-agnostic
    model.multi_label = True # NMS multiple labels per box
    model.max_det = 1000  # maximum number of detections per image

    frame = img.copy()
    results = model(frame)
    predictions = results.pred[0]

    boxes = predictions[:, :4]
    scores = predictions[:, 4]
    classes = predictions[:, 5]
    # Draws Bounding Box onto image
    results.render() 

    # Initialize frame rate calculation
    frame_rate_calc = 30
    freq = cv2.getTickFrequency()

    frame_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    #imW, imH = int(400), int(300)
    imW, imH = int(640), int(640)
    frame_resized = cv2.resize(frame_rgb, (imW, imH))
    input_data = np.expand_dims(frame_resized, axis=0)

    max_score = 0
    max_index = 0
    
    # Loop over all detections and draw detection box if confidence is above minimum threshold
    for i in range(len(scores)):
        curr_score = scores.numpy()
        # Found desired object with decent confidence
        if ((labels[int(classes[i])] == cType.getType()) and (curr_score[i] > max_score) and (curr_score[i] > min_conf_threshold) and (curr_score[i] <= 1.0)):
            # Get bounding box coordinates and draw box
            # Interpreter can return coordinates that are outside of image dimensions, need to force them to be within image using max() and min()
            xmin = int(max(1,(boxes[i][0])))
            ymin = int(max(1,(boxes[i][1])))
            xmax = int(min(imW,(boxes[i][2])))
            ymax = int(min(imH,(boxes[i][3])))

            stop(5)
            print("ball found")



            for cnt in contours_blue:
                contour_area = cv2.contourArea(cnt)
                if contour_area > 1500 and "Blue" not in seen:
                    x, y, w, h = cv2.boundingRect(cnt)
                    cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
                    cv2.putText(img, 'Blue', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
                    print("Blue Ball Found")
                    led.ledIndex(0x01,0,0,255)      #Red
                    led.ledIndex(0x02,0,0,255)    #orange
                    led.ledIndex(0x04,0,0,255)    #yellow
                    led.ledIndex(0x08,0,0,255)      #green
                    led.ledIndex(0x10,0,0,255)    #cyan-blue
                    led.ledIndex(0x20,0,0,255)      #blue
                    led.ledIndex(0x40,0,0,255)    #purple
                    led.ledIndex(0x80,0,0,255)  #white'''
                    seen.append("Blue")


            for cnt in contours_green:
                contour_area = cv2.contourArea(cnt)
                if contour_area > 1500 and "Green" not in seen:
                    x, y, w, h = cv2.boundingRect(cnt)
                    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.putText(img, 'Green', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                    print("Green Ball Found")
                    led.ledIndex(0x01,0,255,0)      #Red
                    led.ledIndex(0x02,0,255,0)    #orange
                    led.ledIndex(0x04,0,255,0)    #yellow
                    led.ledIndex(0x08,0,255,0)      #green
                    led.ledIndex(0x10,0,255,0)    #cyan-blue
                    led.ledIndex(0x20,0,255,0)      #blue
                    led.ledIndex(0x40,0,255,0)    #purple
                    led.ledIndex(0x80,0,255,0)  #white'''
                    seen.append("Green")


            for cnt in contours_yellow:
                contour_area = cv2.contourArea(cnt)
                if contour_area > 1500 and "Yellow" not in seen:
                    x, y, w, h = cv2.boundingRect(cnt)
                    cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
                    cv2.putText(img, 'Blue', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
                    print("Yellow Ball Found")
                    led.ledIndex(0x01,255,255,0)      #Red
                    led.ledIndex(0x02,255,255,0)    #orange
                    led.ledIndex(0x04,255,255,0)    #yellow
                    led.ledIndex(0x08,255,255,0)      #green
                    led.ledIndex(0x10,255,255,0)    #cyan-blue
                    led.ledIndex(0x20,255,255,0)      #bluev
                    led.ledIndex(0x40,255,255,0)   #purple
                    led.ledIndex(0x80,255,255,0)  #white'''
                    seen.append("Yellow")

            for cnt in contours_red:
                contour_area = cv2.contourArea(cnt)
                if contour_area > 1500 and "Red" not in seen:
                    x, y, w, h = cv2.boundingRect(cnt)
                    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
                    cv2.putText(img, 'Red', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                    print("Red Ball Found")
                    led.ledIndex(0x01,255,0,0)      #Red
                    led.ledIndex(0x02,255,0,0)    #orange
                    led.ledIndex(0x04,255,0,0)    #yellow
                    led.ledIndex(0x08,255,0,0)      #green
                    led.ledIndex(0x10,255,0,0)    #cyan-blue
                    led.ledIndex(0x20,255,0,0)      #blue
                    led.ledIndex(0x40,255,0,0)    #purple
                    led.ledIndex(0x80,255,0,0)  #white'''
                    seen.append("Red")



            time.sleep(3)               #wait 3s
            led.colorWipe(led.strip, Color(0,0,0))  #turn off the light
            
                       
            # Draw label
            object_name = labels[int(classes[i])] # Look up object name from "labels" array using class index
            label = '%s: %d%%' % (object_name, int(curr_score[i]*100)) # Example: 'person: 72%'
            labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2) # Get font size
            label_ymin = max(ymin, labelSize[1] + 10) # Make sure not to draw label too close to top of window
            #cv2.rectangle(frame, (xmin,ymin), (xmax,ymax), (10, 255, 0), 2)
            #cv2.rectangle(frame, (xmin, label_ymin-labelSize[1]-10), (xmin+labelSize[0], label_ymin+baseLine-10), (255, 255, 255), cv2.FILLED) # Draw white box to put label text in
            cv2.putText(frame, label, (xmin, label_ymin-7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2) # Draw label text
            #if cType.getType() == "ball":
                
            # Record current max
            max_score = curr_score[i]
            max_index = i

    # Write Image (with bounding box) to file
    cv2.imwrite('video.jpg', frame)
    print("find ball fucntion done")



def turnRight(time1):
    PWM.setMotorModel(2000,2000,-1500,-1500)
    time.sleep(time1)
    PWM.setMotorModel(0,0,0,0)

def forward(distance,time1):
    PWM.setMotorModel(distance,distance,distance,distance)
    time.sleep(time1)
    PWM.setMotorModel(0,0,0,0)

def backward(distance,time1):
    PWM.setMotorModel(-distance,-distance,-distance,-distance)
    time.sleep(time1)
    PWM.setMotorModel(0,0,0,0)

def turnLeft(time1):
    PWM.setMotorModel(-1500,-1500,2000,2000)
    time.sleep(time1)
    PWM.setMotorModel(0,0,0,0)

def moveRight(distance, time1):
    PWM.setMotorModel(distance,-distance,-distance,distance)       #Move right 
    print ("The car is moving right")  
    time.sleep(time1)  

def moveRight(distance, time1):
    PWM.setMotorModel(-distance,distance,distance,-distance)       #Move right 
    print ("The car is moving right")  
    time.sleep(time1)     

def diagonalTopRight(distance, time1):
    PWM.setMotorModel(distance,0,distance,0)       #Move diagonally to the right and forward
    print ("The car is moving diagonally to the right and forward")  
    time.sleep(time1)

def diagonalBottomLeft(distance, time1):
    PWM.setMotorModel(-distance,0,-distance,0)       #Move diagonally to the left and backward
    print ("The car is moving diagonally to the left and backward")  
    time.sleep(time1)

def stop(time1):
    PWM.setMotorModel(0,0,0,0)
    time.sleep(time1)

class Line_Tracking():
    def cameraStuff(self):
        while True:
            picam2 = Picamera2()
            cType.setType("balls")
            picam2.start_and_capture_file('image.jpg', show_preview=False)
            picam2.close()
            print("taking pic rn")
            img = cv2.imread('image.jpg')
            find_ball(img)



    def __init__(self):
        self.IR01 = 14
        self.IR02 = 15
        self.IR03 = 23
        GPIO.setmode(GPIO.BCM)
        GPIO.setup(self.IR01,GPIO.IN)
        GPIO.setup(self.IR02,GPIO.IN)
        GPIO.setup(self.IR03,GPIO.IN)
        cType.setType("balls")



    def run(self):
        while True:
            # PWM.setMotorModel(700,700,700,700)
            #os.system('libcamera-still -o image.jpg')

            
            # picam2.start_and_capture_file('image.jpg', show_preview=False)
            # picam2.close()
            # print("taking pic rn")

            # img = cv2.imread('image.jpg')
            # find_ball(img)

            
            
            PWM.setMotorModel(700,700,700,700)
            self.LMR=0x00
            print("idk why ")
            if GPIO.input(self.IR01)==True:
                self.LMR=(self.LMR | 4)
            if GPIO.input(self.IR02)==True:
                self.LMR=(self.LMR | 2)
            if GPIO.input(self.IR03)==True:
                self.LMR=(self.LMR | 1)
            if self.LMR==7: #Checks if the 2nd sensor detects a line.
                stop(.25)
                backward(700,.6)
                turnRight(.12)
            elif self.LMR==6: #Checks if only the 1st sensor detects a line.
                stop(.25)
                turnRight(.12)
            elif self.LMR==4: # Checks if the 1st and 2nd sensors detect a line.
                stop(.25)
                turnRight(.12)
            elif self.LMR==1: # Checks if only the 3rd sensor detects a line.
                stop(.25)
                turnLeft(.12)
            elif self.LMR==3: #Checks if the 2nd and 3rd sensors detect a line.
                stop(.25)
                turnLeft(.12)

            else:
                continue

            print(self.LMR)
            
            

# Main program logic follows:
infared = Line_Tracking()
if __name__ == '__main__':
    print ('Program is starting ... ')
    try:
        t1 = threading.Thread(target=infared.run())
        t2 = threading.Thread(target=infared.cameraStuff())
        t2.start()
        t1.start()
        t1.join()
        t2.join()

    except KeyboardInterrupt:  # When 'Ctrl+C' is pressed, the child program  will be  executed.
        PWM.setMotorModel(0,0,0,0)
