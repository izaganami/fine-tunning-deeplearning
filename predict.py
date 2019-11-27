from keras.models import load_model
from collections import deque
import numpy as np
import argparse
import pickle
import cv2

ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", required=True, help="path to trained serialized model")
ap.add_argument("-l", "--label-bin", required=True, help="path to  label binarizer")
ap.add_argument("-i", "--input", required=True, help="path to our input video")
ap.add_argument("-o", "--output", required=True, help="path to our output video")
ap.add_argument("-s", "--size", type=int, default=128, help="size of queue for averaging")
args = vars(ap.parse_args())

# this is a test running file i ll finish it after training our model (changing softmax values to be precise xD )
