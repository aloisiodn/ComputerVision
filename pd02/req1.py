#!/usr/bin/python3.6
# Requisito1
# Author: Aloisio Dourado Neto
# Discipline: Computer Vision/UnB
# Professor: Teo de Campos
# Mail: aloisio.dourado.bh@gmail.com
# Created Time: Sat 17 Mar 2018

import sys, os
import cv2
import numpy as np
import math

PROJECT_NAME = "Computer Vision Demonstration Project 2 - Requirement 1"
PROJECT_DESC = "Distance in pixels"
CIRCLE_RADIUS = 7
CIRCLE_COLOR = (0,0,255)
RULLER_COLOR = (255,0,0)

click_count = 0
p1 = (0,0)
p2 = (0,0)
click_count = 0


# Function that effectivaly to the project job
def do_the_job():


    global click_count
    global p1, p2


    def euclidian_dist(p1, p2):

        return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

    # Mouse Event Handler
    def on_mouse_event(event, mouse_col, mouse_lin, flags, param):
        # grab references to the global variables
        global click_count
        global p1, p2

        # if the left mouse button was clicked, show coordinates and B G R color
        if event == cv2.EVENT_LBUTTONDOWN:

            click_count += 1
            if click_count == 3:
                click_count = 1

            if click_count == 1:
                p1 = (mouse_col, mouse_lin)

            if click_count == 2:
                p2 = (mouse_col, mouse_lin)
                print("\nDistance=%.2f" % euclidian_dist(p1, p2))

    win_name = PROJECT_NAME
    cv2.namedWindow(win_name)  # Create a named window
    cv2.moveWindow(win_name, 40, 30)  # Move it to (40,30)

    cv2.setMouseCallback(win_name, on_mouse_event)

    exit = False

    while not exit:
        cap = cv2.VideoCapture(0)
        while cap.isOpened():
            # Capture frame-by-frame
            ret, f = cap.read()

            frame = cv2.flip(f, 1)

            if ret:

                if click_count > 0:
                    cv2.circle(frame, p1, CIRCLE_RADIUS, CIRCLE_COLOR, thickness=2, lineType=8, shift=0)

                if click_count > 1:
                    cv2.circle(frame, p2, CIRCLE_RADIUS, CIRCLE_COLOR, thickness=2, lineType=8, shift=0)
                    cv2.line(frame, p1, p2, RULLER_COLOR, thickness=2, lineType=8, shift=0)

                # Display the resulting frame
                cv2.imshow(win_name, frame)

                if cv2.waitKey(3)& 0xFF == ord('q'):

                    exit = True
                    break
            else:
                break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()



# Main Function
def Run():

    print("\n%s\n%s" % (PROJECT_NAME, PROJECT_DESC))
    print("\nTested with python3.6.4, opencv 3 and opencv-python 3.4.0")

    print("\nCurrent opencv-python version: %s\n" % cv2.__version__)

    if len(sys.argv) != 1:
        print('\nSyntax: %s\n' % sys.argv[0])
        sys.exit(-1)

    print("press q to end...")

    do_the_job()

if __name__ == '__main__':
  Run()