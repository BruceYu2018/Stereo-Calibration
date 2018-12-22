import cv2 as cv
import numpy as np

stereoParams = np.load('stereoParams.npz')
M1 = stereoParams['M1']
M2 = stereoParams['M2']
D1 = stereoParams['D1']
D2 = stereoParams['D2']
R1 = stereoParams['R1']
R2 = stereoParams['R2']
P1 = stereoParams['P1']
P2 = stereoParams['P2']
ROI1 = stereoParams['ROI1']
ROI2 = stereoParams['ROI2']
imagesize = (1280, 1024)
#imagesize = (1920, 1080)

leftmaps = cv.initUndistortRectifyMap(M1, D1, R1, P1, imagesize, cv.CV_16SC2)
rightmaps = cv.initUndistortRectifyMap(M2, D2, R2, P2, imagesize, cv.CV_16SC2)
y1 = max(ROI1[0], ROI2[0])
x1 = max(ROI1[1], ROI2[1])
y2 = min(ROI1[0] + ROI1[2], ROI2[0] + ROI2[2])
x2 = min(ROI1[1] + ROI1[3], ROI2[1] + ROI2[3])
newsize = (y2-y1, x2-x1)

videopath = "/home/zhenyu/PycharmProjects/stereo/CalibVideo1.avi"
video = cv.VideoCapture(videopath)
outfile = 'rectifiedVideo.avi'
writer = cv.VideoWriter(outfile, cv.VideoWriter_fourcc(*'DIVX'), 10, (newsize[0]*2, newsize[1]))
rval = True
while rval:
    rval, frame = video.read()
    mid = frame.shape[1]/2
    left = frame[:, :mid]
    right = frame[:, mid:]
    img_l_remap = cv.remap(left, leftmaps[0], leftmaps[1], cv.INTER_LINEAR)
    img_r_remap = cv.remap(right, rightmaps[0], rightmaps[1], cv.INTER_LINEAR)
    imgl = img_l_remap[x1:x2, y1:y2, :]
    imgr = img_r_remap[x1:x2, y1:y2, :]
    print imgl.shape
    #imgl = cv.resize(imgl, imagesize, 0, 0, cv.INTER_LINEAR)
    #imgr = cv.resize(imgr, imagesize, 0, 0, cv.INTER_LINEAR)
    res = np.concatenate((imgl, imgr), 1)

    cv.namedWindow("video", cv.WINDOW_NORMAL)
    cv.imshow("video", frame)
    cv.namedWindow("res", cv.WINDOW_NORMAL)
    cv.imshow("res", res)
    writer.write(res)
    cv.waitKey(1)

video.release()
