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
import pandas as pd
import math



PROJECT_NAME = "Computer Vision Demonstration Project 2 - Requirement 2"
PROJECT_DESC = "Calibration"
SNAPSHOT_WIN = "Snapshot"
UNDISTORT_WIN = "Undistort"
RAW_WIN = "RAW Video"

CIRCLE_RADIUS = 5
CIRCLE_COLOR = (0,0,255)
RULLER_COLOR = (255,0,0)

raw_click_count = 0
raw_p1 = (0,0)
raw_p2 = (0,0)

und_click_count = 0
und_p1 = (0,0)
und_p2 = (0,0)



# Function that effectivaly to the project job
def calibrate_round():

    print("adjust camera and chess board and press s to start calibrating...")
    start = False


    global click_count
    global p1, p2

    n_spanspots = 10
    spanspot_count = 0

    frame_step = 30
    frame_wait = 30

    board_w = 8;
    board_h = 6;
    objpoints = []  # 3d point in real world space
    imgpoints = []  # 2d points in image plane.
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    objp = np.zeros((board_h * board_w, 3), np.float32)
    objp[:, :2] = np.mgrid[0:board_w, 0:board_h].T.reshape(-1, 2)

    cv2.namedWindow(SNAPSHOT_WIN)  # Create a named window
    cv2.moveWindow(SNAPSHOT_WIN, 10, 10)
    cv2.namedWindow(RAW_WIN)  # Create a named window
    cv2.moveWindow(RAW_WIN, 650, 10)

    cap = cv2.VideoCapture(0)
    while spanspot_count < n_spanspots:
        if cap.isOpened():


            # Capture frame-by-frame
            ret, frame = cap.read()

            if ret:

                #frame = cv2.flip(f, 1)

                if frame_wait > 0:
                    frame_wait -= 1
                else:
                    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                    found, corners = cv2.findChessboardCorners(frame_gray, (board_w, board_h), None)
                    if start and found and (len(corners) == board_w * board_h):


                        frame_wait = frame_step

                        spanspot_count += 1

                        print("found spanspot %d of %d" % (spanspot_count, n_spanspots))

                        objpoints.append(objp)

                        corners2 = cv2.cornerSubPix(frame_gray, corners, (11, 11), (-1, -1), criteria)
                        imgpoints.append(corners2)

                        # Draw and display the corners
                        frame = cv2.drawChessboardCorners(frame, (board_w, board_h), corners2, ret)

                        # Display the resulting frame
                        cv2.imshow(SNAPSHOT_WIN, frame) # Se achou mostra snapshot colorido

                    else:
                        cv2.imshow(SNAPSHOT_WIN, frame_gray) # Se não achou mostra snapshot cinza


                # Display the resulting frame
                cv2.imshow(RAW_WIN, frame)

            if cv2.waitKey(3) & 0xFF == ord('s'):
                start = True

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, frame_gray.shape[::-1],None, None)

    return mtx, dist, rvecs, tvecs

def calibrate():

    intrinsics_df = pd.DataFrame()
    distortion_df = pd.DataFrame()


    for round in range(5):
        print("\nCalibration round %d #################################" % (round+1))
        intrinsics, distortion, rvecs, tvecs = calibrate_round()
        print("\nintrinsic parameters:")
        print(intrinsics)
        print("\nrvecs:")
        print(rvecs)

        print("\ntvecs:")
        print(tvecs)

        intrinsics = np.array(intrinsics).reshape(1,9)
        distortion = np.array(distortion).reshape(1,5)

        intrinsics_df = intrinsics_df.append(pd.DataFrame(data=intrinsics,
                                                          columns=['p1', 'p2', 'p3',
                                                                   'p4', 'p5', 'p6',
                                                                   'p7', 'p8', 'p9'])).reset_index(drop=True)
        distortion_df = distortion_df.append(pd.DataFrame(data=distortion,
                                                          columns=['p1', 'p2', 'p3', 'p4', 'p5'])).reset_index(drop=True)

    intrinsics = np.array(intrinsics_df.mean()).reshape(3,3)
    distortion = np.array(distortion_df.mean()).reshape(1,5)

    intrinsics_std = np.array(intrinsics_df.std()).reshape(3,3)
    distortion_std = np.array(distortion_df.std()).reshape(1,5)


    print("\nFinal intrinsic parameters ###########################")
    print("mean:")
    print(intrinsics)
    print("std:")
    print(intrinsics_std)

    print("\nFinal distortion parameters #########################:")
    print("mean:")
    print(distortion)
    print("std:")
    print(distortion_std)

    print("\nWriting xml files #########################:")
    fs = cv2.FileStorage('intrinsics.xml', cv2.FILE_STORAGE_WRITE)
    fs.write("floatdata", intrinsics)
    fs.release()

    fs = cv2.FileStorage('distortion.xml', cv2.FILE_STORAGE_WRITE)
    fs.write("floatdata", distortion)
    fs.release()

    return intrinsics, distortion


def measure(intrinsics, distortion):

    global raw_click_count
    global raw_p1, raw_p2

    global und_click_count
    global und_p1, und_p2


    print("\nclick 2 points in the windows to measure the distance...")

    print("\npress q to end...")




    def euclidian_dist(p1, p2):

        return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

    # Mouse Event Handler
    def on_mouse_event(event, mouse_col, mouse_lin, click_count, p1, p2 ):
        # grab references to the global variables

        # if the left mouse button was clicked, show coordinates and B G R color
        if event == cv2.EVENT_LBUTTONDOWN:

            click_count += 1
            if click_count == 3:
                click_count = 1

            if click_count == 1:
                p1 = (mouse_col, mouse_lin)

            if click_count == 2:
                p2 = (mouse_col, mouse_lin)
                print("Dist=%.2f" % euclidian_dist(p1, p2))

        return  click_count, p1, p2

    def raw_on_mouse_event(event, mouse_col, mouse_lin, flags, param):
        # grab references to the global variables
        global raw_click_count
        global raw_p1, raw_p2

        raw_click_count, raw_p1, raw_p2 = on_mouse_event(event, mouse_col, mouse_lin, raw_click_count, raw_p1, raw_p2)

    def und_on_mouse_event(event, mouse_col, mouse_lin, flags, param):
        # grab references to the global variables
        global und_click_count
        global und_p1, und_p2

        und_click_count, und_p1, und_p2 = on_mouse_event(event, mouse_col, mouse_lin, und_click_count, und_p1, und_p2)


    def window_measure(win_name, frame, click_count, p1, p2):
        if click_count > 0:
            cv2.circle(frame, p1, CIRCLE_RADIUS, CIRCLE_COLOR, thickness=2, lineType=8, shift=0)

        if click_count > 1:
            cv2.circle(frame, p2, CIRCLE_RADIUS, CIRCLE_COLOR, thickness=2, lineType=8, shift=0)
            cv2.line(frame, p1, p2, RULLER_COLOR, thickness=2, lineType=8, shift=0)

        # Display the resulting frame
        cv2.imshow(win_name, frame)

    cap = cv2.VideoCapture(0)
    ret, f = cap.read()
    h, w = f.shape[:2]

    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(intrinsics, distortion, (w, h), 1, (w, h))

    mapx, mapy = cv2.initUndistortRectifyMap(intrinsics, distortion, None, newcameramtx, (w, h), 5)

    cv2.namedWindow(UNDISTORT_WIN)  # Create a named window
    cv2.moveWindow(UNDISTORT_WIN, 10, 10)
    cv2.namedWindow(RAW_WIN)  # Create a named window
    cv2.moveWindow(RAW_WIN, 650, 10)

    cv2.setMouseCallback(UNDISTORT_WIN, und_on_mouse_event)
    cv2.setMouseCallback(RAW_WIN, raw_on_mouse_event)

    #cap = cv2.VideoCapture(0)
    exit = False
    while not exit:

        # Capture frame-by-frame
        ret, frame = cap.read()

        if ret:

            #frame = cv2.flip(f, 1)

            undist = cv2.remap(frame, mapx, mapy, cv2.INTER_LINEAR)

            window_measure(UNDISTORT_WIN, undist, und_click_count, und_p1, und_p2)
            window_measure(RAW_WIN, frame, raw_click_count, raw_p1, raw_p2)

            if cv2.waitKey(3)& 0xFF == ord('q'):
                exit = True

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()



def do_the_job():
    intrinsics, distortion = calibrate()
    measure(intrinsics, distortion)

# Main Function
def Run():

    print("\n%s\n%s" % (PROJECT_NAME, PROJECT_DESC))
    print("\nTested with python3.6.4, opencv 3 and opencv-python 3.4.0")

    print("\nCurrent opencv-python version: %s\n" % cv2.__version__)

    if len(sys.argv) != 1:
        print('\nSyntax: %s\n' % sys.argv[0])
        sys.exit(-1)

    do_the_job()

if __name__ == '__main__':
  Run()