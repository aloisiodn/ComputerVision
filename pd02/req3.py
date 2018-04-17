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
from scipy.optimize import least_squares
import scipy.linalg as linalg


PROJECT_NAME = "Computer Vision Demonstration Project 2 - Requirement 3"
PROJECT_DESC = "Extrisic Calibration"
SNAPSHOT_WIN = "Snapshot"
UNDISTORT_WIN = "Undistort"
RAW_WIN = "RAW Video"

CIRCLE_RADIUS = 5
CIRCLE_COLOR = (0,0,255)
RULLER_COLOR = (255,0,0)

BOARD_W = 8;
BOARD_H = 6;

raw_click_count = 0
raw_p1 = (0,0)
raw_p2 = (0,0)

und_click_count = 0
und_p1 = (0,0)
und_p2 = (0,0)





def calibrate():

    width = 20.4 / BOARD_W
    heigth = 14.6 / BOARD_H

    WRL = np.zeros((48,2))

    i = 0
    for y in range(BOARD_H):
        for x in range(BOARD_W):
            WRL[i] = ((x * width), (y * heigth))
            i += 1

    fs = cv2.FileStorage('intrinsics.xml', cv2.FILE_STORAGE_READ)
    intrinsics = fs.getNode('floatdata').mat()
    fs.release()

    fs = cv2.FileStorage('distortion.xml', cv2.FILE_STORAGE_READ)
    distortion = fs.getNode('floatdata').mat()
    fs.release()

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    h, w = frame.shape[:2]

    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(intrinsics, distortion, (w, h), 1, (w, h))

    mapx, mapy = cv2.initUndistortRectifyMap(intrinsics, distortion, None, newcameramtx, (w, h), 5)

    cv2.namedWindow(UNDISTORT_WIN)  # Create a named window
    cv2.moveWindow(UNDISTORT_WIN, 10, 10)
    cv2.namedWindow(RAW_WIN)  # Create a named window
    cv2.moveWindow(RAW_WIN, 650, 10)

    found = False
    exit = False
    start = False

    print("setup camera and chessboard and press s to start...")
    print("press q to quit without finding chessboard...")

    while (not found) and (not exit):

        # Capture frame-by-frame
        ret, frame = cap.read()

        if ret:

            #frame = cv2.flip(f, 1)

            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            undist = cv2.remap(frame, mapx, mapy, cv2.INTER_LINEAR)

            undist_gray = cv2.cvtColor(undist, cv2.COLOR_BGR2GRAY)

            ret, corners = cv2.findChessboardCorners(undist_gray, (BOARD_W, BOARD_H), None)

            if start and ret and (len(corners) == BOARD_W * BOARD_H):

                found = True

                CAM = cv2.cornerSubPix(undist_gray, corners, (11, 11), (-1, -1), criteria)

                # Draw and display the corners
                undist = cv2.drawChessboardCorners(undist, (BOARD_W, BOARD_H), CAM, ret)

                # Display the resulting frame
                cv2.imshow(UNDISTORT_WIN, undist)  # Se achou mostra snapshot colorido
            else:
                cv2.imshow(UNDISTORT_WIN, undist_gray)  # Se não achou mostra snapshot p&b

            # Display the resulting frame
            cv2.imshow(RAW_WIN, frame)

            key = cv2.waitKey(3)

            if key& 0xFF == ord('s'):
                start = True
            if key & 0xFF == ord('q'):
                return

    CAM = CAM.reshape(48,2)

    A = np.zeros((2*BOARD_W*BOARD_H, 9))

    for i in range(BOARD_W*BOARD_H):
        A[i*2] = (
                  WRL[i][0],                # X
                  WRL[i][1],                # Y
                  1,                        # 1
                  0,                        # 0
                  0,                        # 0
                  0,                        # 0
                  - CAM[i][0] * WRL[i][0],  # -xX
                  - CAM[i][0] * WRL[i][1],  # -xY
                  - CAM[i][0]                 # x
                  )
        A[(i*2)+1] = (
                  0,                        # 0
                  0,                        # 0
                  0,                        # 0
                  WRL[i][0],                # X
                  WRL[i][1],                # Y
                  1,                        # 1
                  - CAM[i][1] * WRL[i][0],  # -yX
                  - CAM[i][1] * WRL[i][1],  # -yY
                  - CAM[i][1]                 # y
                  )

    #np.set_printoptions(linewidth=160)

    U, s, Vh = linalg.svd(A)

    V = np.transpose(Vh)

    P = np.array([row[8] for row in V]).reshape((3,3))

    p = P.reshape(9)

    X = np.array([row[0] for row in WRL])
    Y = np.array([row[1] for row in WRL])

    x = np.array([row[0] for row in CAM])
    y = np.array([row[1] for row in CAM])

    def func(p, x, y, X, Y):

        proj_x = (p[0]*X + p[1]*Y + p[2]) / (p[6]*X + p[7]*Y + p[8])

        proj_y = (p[3]*X + p[4]*Y + p[5]) / (p[6]*X + p[7]*Y + p[8])

        r = (x - proj_x) ** 2 + (y - proj_y) ** 2

        return r

    res = least_squares(func, p, args=(np.array(x), np.array(y), np.array(X), np.array(Y)), max_nfev=9000, verbose=0)

    new_P = res.x.reshape((3, 3))

    H = np.array(linalg.inv(intrinsics) @ new_P)

    h1 = np.array([row[0] for row in H])
    h2 = np.array([row[1] for row in H])

    _R = (2/(linalg.norm(h1) + linalg.norm(h2))) * H

    t = np.array([row[2] for row in _R])

    r1 = np.array([row[0] for row in _R])
    r2 = np.array([row[1] for row in _R])
    r3 = r1 * r2

    _R = np.append(np.append(r1, r2), r3).reshape((3,3))

    U, s, Vh = linalg.svd(_R)

    R = U @ Vh

    r = np.array([row[:2] for row in R])
    print("\nr (rotation matrix:")
    print(r)

    print("\nt (translation vector:")
    print(t)
    print("\n||t||:", linalg.norm(t))

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()
    cv2.namedWindow(UNDISTORT_WIN)  # Create a named window
    cv2.moveWindow(UNDISTORT_WIN, 10, 10)
    cv2.namedWindow(RAW_WIN)  # Create a named window
    cv2.moveWindow(RAW_WIN, 650, 10)


def do_the_job():
    calibrate()

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