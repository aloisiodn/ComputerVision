#!/usr/bin/python3.6
# Demonstration Project 4
# Author: Aloisio Dourado Neto
# Discipline: Computer Vision/UnB
# Professor: Teo de Campos
# Mail: aloisio.dourado.bh@gmail.com
# Created Time: Sat 17 Mar 2018

import sys, os
import cv2
import numpy as np
import math

PROJECT_NAME = "Computer Vision Demonstration Project 4"
PROJECT_DESC = "Pixel selection based on color on webcam"
THRESHOLD = 13
MASK_BGR_COLOR = (0, 0, 255)

click = False
color = [0 ,0, 0]


# Function that effectivaly to the project job
def do_the_job():


    global click
    global color


    def euclidian_mask(image, color, threshold):

        np_img = np.array(image, dtype='float32')
        np_point = np.array(color)

        b_sq, g_sq, r_sq = cv2.split((np_img - np_point) ** 2)

        return np.sqrt(b_sq + g_sq + r_sq) < threshold

    def apply_mask(image, mask, bgr_color):

        b, g, r = cv2.split(image)

        b[mask] = bgr_color[0]
        g[mask] = bgr_color[1]
        r[mask] = bgr_color[2]

        return cv2.merge((b, g, r))

    # Mouse Event Handler
    def on_mouse_event(event, mouse_col, mouse_lin, flags, param):
        # grab references to the global variables
        global click
        global color

        # if the left mouse button was clicked, show coordinates and B G R color
        if event == cv2.EVENT_LBUTTONDOWN:

            click = True
            coord = (mouse_lin, mouse_col)
            color = frame[mouse_lin][mouse_col]
            print("Cor [B G R] do pixel", coord, ": ", color)


    global img, threshold

    win_name = PROJECT_NAME
    cv2.namedWindow(win_name)  # Create a named window
    cv2.moveWindow(win_name, 40, 30)  # Move it to (40,30)

    cv2.setMouseCallback(win_name, on_mouse_event)

    exit = False

    while(not exit):
        cap = cv2.VideoCapture(0)
        while (cap.isOpened()):
            # Capture frame-by-frame
            ret, f = cap.read()

            frame = cv2.flip(f,1)

            if ret:

                if click:
                    mask = euclidian_mask(frame, color, THRESHOLD)
                    frame = apply_mask(frame, mask, MASK_BGR_COLOR)

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
    global THRESHOLD

    print("\n%s\n%s" % (PROJECT_NAME, PROJECT_DESC))
    print("\nTested with python3.5, opencv 3 and opencv-python 3.1.0")

    print("\nCurrent opencv-python version: %s\n" % cv2.__version__)

    if len(sys.argv) < 1:
        print('\nSyntax: %s <color_threshold> (default to 13)\n' % sys.argv[0])
        sys.exit(-1)

    if len(sys.argv) == 2:
        try:
            THRESHOLD =  float(sys.argv[1])
        except:
            print('\nInvalid value for color_threshold:',sys.argv[1] )
            sys.exit(-1)

    print("Color threshold set to", THRESHOLD)
    print("press q to end...")

    do_the_job()

#if __name__ == '__main__':
Run()