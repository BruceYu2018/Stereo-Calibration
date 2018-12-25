import numpy as np
import cv2 as cv
import os
import time

leftcam_filepath = os.path.join(os.getcwd(), "left_imgs")
rightcam_filepath = os.path.join(os.getcwd(), "right_imgs")
chessboardRow = 6
chessboardCol = 9
squareSize = 24.23  # unit:mm


'''1. Get the object points and image points'''
# Set virtual object point
objp = np.zeros((chessboardRow*chessboardCol, 3), np.float32)
objp[:, :2] = np.mgrid[0:chessboardRow, 0:chessboardCol].T.reshape(-1, 2)
objp = objp*squareSize

objpts = []
imgpts_leftcam = []
imgpts_rightcam = []

leftcam_img = os.listdir(leftcam_filepath)
leftcam_img.sort()
rightcam_img = os.listdir(rightcam_filepath)
rightcam_img.sort()

start_time = time.time()
count = 0
flag = cv.CALIB_CB_ADAPTIVE_THRESH + cv.CALIB_CB_NORMALIZE_IMAGE + cv.CALIB_CB_FAST_CHECK
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.01)

print("Chessboard detection start.")
for imgname_l, imgname_r in zip(leftcam_img, rightcam_img):
     img_l = cv.imread(os.path.join(leftcam_filepath, imgname_l))
     gray_l = cv.cvtColor(img_l, cv.COLOR_BGR2GRAY)
     ret_l, corners_l = cv.findChessboardCorners(gray_l, (chessboardRow, chessboardCol), flag)

     img_r = cv.imread(os.path.join(rightcam_filepath, imgname_r))
     gray_r = cv.cvtColor(img_r, cv.COLOR_BGR2GRAY)
     ret_r, corners_r = cv.findChessboardCorners(gray_r, (chessboardRow, chessboardCol), flag)

     if ret_l is True and ret_r is True:
         objpts.append(objp)
         corners_refined_l = cv.cornerSubPix(gray_l, corners_l, (11, 11), (-1, -1), criteria)
         imgpts_leftcam.append(corners_refined_l)

         corners_refined_r = cv.cornerSubPix(gray_r, corners_r, (11, 11), (-1, -1), criteria)
         imgpts_rightcam.append(corners_refined_r)

         count += 1

print (str(count) + " pairs successfully detected.")


'''2. Get single camera intrinsic matrix and distortion coefficients'''
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 100, 1e-05)
flag = cv.CALIB_FIX_ASPECT_RATIO + cv.CALIB_ZERO_TANGENT_DIST + cv.CALIB_USE_INTRINSIC_GUESS + \
       cv.CALIB_FIX_K3 + cv.CALIB_FIX_K4 + cv.CALIB_FIX_K5
print("Calibrate left camera.")
retval1 = cv.initCameraMatrix2D(objpts, imgpts_leftcam, gray_l.shape[::-1], 0)
ret_l, mtx_l, dist_l, rvecs_l, tvecs_l = cv.calibrateCamera(objpts, imgpts_leftcam, gray_l.shape[::-1], retval1,
                                                            None, flags=flag, criteria=criteria)
print("Calibrate right camera.")
retval2 = cv.initCameraMatrix2D(objpts, imgpts_leftcam, gray_l.shape[::-1], 0)
ret_r, mtx_r, dist_r, rvecs_r, tvecs_r = cv.calibrateCamera(objpts, imgpts_rightcam, gray_r.shape[::-1], retval2,
                                                            None, flags=flag, criteria=criteria)
'''
# Check the single camera calibration result, left here
maps = cv.initUndistortRectifyMap(mtx_l, dist_l, None, mtx_l, gray_l.shape[::-1], cv.CV_16SC2)
for imgname_l in leftcam_img:
    img_l = cv.imread(leftcam_filepath + '/' + imgname_l)
    dst = cv.remap(img_l, maps[0], maps[1], cv.INTER_LINEAR)
    cv.imshow('imgl', img_l)
    cv.imshow('res', dst)
    cv.waitKey(0)
'''


'''3. Start the stereo calibration, compute the corresponding parameters, get the 
      rectifying parameters, saving the parameters, and rectified results.'''
print("Stereo calibration.")
flag = cv.CALIB_USE_INTRINSIC_GUESS + cv.CALIB_SAME_FOCAL_LENGTH + cv.CALIB_FIX_ASPECT_RATIO + cv.CALIB_ZERO_TANGENT_DIST
retval, cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2,  R, T, E, F = cv.stereoCalibrate(objpts,
                          imgpts_leftcam, imgpts_rightcam, mtx_l, dist_l, mtx_r, dist_r, gray_l.shape[::-1],
                          flags=flag, criteria=criteria)

R1, R2, P1, P2, Q, validPixROI1, validPixROI2 = cv.stereoRectify(cameraMatrix1, distCoeffs1, cameraMatrix2,
                distCoeffs2, gray_l.shape[::-1], R, T, flags=cv.CALIB_ZERO_DISPARITY, alpha=0, newImageSize=(0, 0))

# save the rectified parameters
np.savez('stereoParams.npz', M1=cameraMatrix1, M2=cameraMatrix2, D1=distCoeffs1, D2=distCoeffs2,
         R1=R1, R2=R2, P1=P1, P2=P2, ROI1=validPixROI1, ROI2=validPixROI2)

end_time = time.time()
print("Elapsed time : " + str(end_time - start_time) + " s")

# test and show the result
leftmaps = cv.initUndistortRectifyMap(cameraMatrix1, distCoeffs1, R1, P1, gray_l.shape[::-1], cv.CV_16SC2)
rightmaps = cv.initUndistortRectifyMap(cameraMatrix2, distCoeffs2, R2, P2, gray_l.shape[::-1], cv.CV_16SC2)
# get the valid region manually
y1 = max(validPixROI1[0], validPixROI2[0])
x1 = max(validPixROI1[1], validPixROI2[1])
y2 = min(validPixROI1[0] + validPixROI1[2], validPixROI2[0] + validPixROI2[2])
x2 = min(validPixROI1[1] + validPixROI1[3], validPixROI2[1] + validPixROI2[3])

flag = cv.CALIB_CB_ADAPTIVE_THRESH + cv.CALIB_CB_NORMALIZE_IMAGE + cv.CALIB_CB_FAST_CHECK
for imgname_l, imgname_r, imgpts_l in zip(leftcam_img, rightcam_img, imgpts_leftcam):
    img_l = cv.imread(os.path.join(leftcam_filepath, imgname_l))
    img_r = cv.imread(os.path.join(rightcam_filepath, imgname_r))
    img_l_remap = cv.remap(img_l, leftmaps[0], leftmaps[1], cv.INTER_LINEAR)
    img_r_remap = cv.remap(img_r, rightmaps[0], rightmaps[1], cv.INTER_LINEAR)
    #imgl = img_l_remap[x1:x2, y1:y2, :]
    #imgr = img_r_remap[x1:x2, y1:y2, :]
    #cv.rectangle(img_l_remap, (validPixROI1[0], validPixROI1[1]), (validPixROI1[0]+validPixROI1[2], validPixROI1[1] + validPixROI1[3]), (0, 0, 255))
    #cv.rectangle(img_r_remap, (validPixROI2[0], validPixROI2[1]), (validPixROI2[0]+validPixROI2[2], validPixROI2[1] + validPixROI2[3]), (0, 0, 255))
    full_image_original = np.concatenate((img_l, img_r), 1)
    full_image = np.concatenate((img_l_remap, img_r_remap), 1)

    # draw the check line
    cv.circle(full_image_original, tuple(imgpts_l[0][0]), 3, (0, 0, 255), 1)
    cv.line(full_image_original, (0, int(imgpts_l[0][0][1])),
            (full_image_original.shape[1], int(imgpts_l[0][0][1])), (0, 0, 255), 1)
    gray_l_remap = cv.cvtColor(img_l_remap, cv.COLOR_BGR2GRAY)
    ret_l, corners_l = cv.findChessboardCorners(gray_l_remap, (chessboardRow, chessboardCol), flag)
    gray_r_remap = cv.cvtColor(img_r_remap, cv.COLOR_BGR2GRAY)
    ret_r, corners_r = cv.findChessboardCorners(gray_r_remap, (chessboardRow, chessboardCol), flag)
    if ret_l is True and ret_r is True:
        cv.circle(full_image, tuple(corners_l[0][0]), 3, (0, 0, 255), 1)
        cv.circle(full_image, (int(corners_r[0][0][0]) + img_l.shape[1], int(corners_r[0][0][1])), 3, (255, 0, 0), 1)
        cv.line(full_image, (0, int(corners_l[0][0][1])),
                (full_image.shape[1], int(corners_l[0][0][1])), (0, 0, 255), 1)

    result_image = np.concatenate((full_image_original, full_image), 0)

    cv.namedWindow('res', cv.WINDOW_NORMAL)
    cv.imshow('res', result_image)
    cv.waitKey(0)