import time
from Motor import *
import RPi.GPIO as GPIO


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

class Line_Tracking:
    def __init__(self):
        self.IR01 = 14
        self.IR02 = 15
        self.IR03 = 23
        GPIO.setmode(GPIO.BCM)
        GPIO.setup(self.IR01,GPIO.IN)
        GPIO.setup(self.IR02,GPIO.IN)
        GPIO.setup(self.IR03,GPIO.IN)
    def run(self):
        while True:
            PWM.setMotorModel(700,700,700,700)
            self.LMR=0x00
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
            
            
infrared=Line_Tracking()
# Main program logic follows:
if __name__ == '__main__':
    print ('Program is starting ... ')
    try:
        infrared.run()
    except KeyboardInterrupt:  # When 'Ctrl+C' is pressed, the child program  will be  executed.
        PWM.setMotorModel(0,0,0,0)
