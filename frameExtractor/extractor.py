import sys
import argparse

import os
import cv2

def extractImages(pathIn, pathOut):
    if not os.path.exists(pathOut):
        os.makedirs(pathOut)
    count = 0
    vidcap = cv2.VideoCapture(pathIn)
    success, image = vidcap.read()
    while success:
        cv2.imwrite(pathOut + "\\frame%d.jpg" % count, image)     # save frame as JPEG file
        vidcap.set(cv2.CAP_PROP_POS_MSEC, (count*1000))    # added this line
        success, image = vidcap.read()
        if count % 100 == 0:
            print('Read a new frame: ', success)
        count = count + 1


vids = ['JujutsuKaisen-S02E02','JujutsuKaisen-S02E03','JujutsuKaisen-S02E04','JujutsuKaisen-S02E05',
        'JujutsuKaisen-S02E06','JujutsuKaisen-S02E07','JujutsuKaisen-S02E08','JujutsuKaisen-S02E09',
        'JujutsuKaisen-S02E10','JujutsuKaisen-S02E11','JujutsuKaisen-S02E12','JujutsuKaisen-S02E13',
        'JujutsuKaisen-S02E14','JujutsuKaisen-S02E15','JujutsuKaisen-S02E16','JujutsuKaisen-S02E17',
        'JujutsuKaisen-S02E18','JujutsuKaisen-S02E19','JujutsuKaisen-S02E20','JujutsuKaisen-S02E21',
        'JujutsuKaisen-S02E22','JujutsuKaisen-S02E23']

# extractImages('D:/anime/jjk/JujutsuKaisen-s2/JujutsuKaisen-S02E01.mkv', './dataset/jjk-s2e01/')
if __name__ == "__main__":
    for vid in vids:
        extractImages(f'D:/anime/jjk/JujutsuKaisen-s2/{vid}.mkv', f'./dataset/{vid}/')