import sys
import argparse

import cv2
print(cv2.__version__)

def extractImages(pathIn, pathOut):
    count = 0
    vidcap = cv2.VideoCapture(pathIn)
    success, image = vidcap.read()
    while success:
        cv2.imwrite( pathOut + "\\frame%d.jpg" % count, image)     # save frame as JPEG file
        vidcap.set(cv2.CAP_PROP_POS_MSEC,(count*1000))    # added this line
        success, image = vidcap.read()
        print('Read a new frame: ', success)
        count = count + 1


extractImages('./vid.mp4', './')