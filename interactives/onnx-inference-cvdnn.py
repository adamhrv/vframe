# %% markdown
# # Scratchpad temporary code

# %% codecell
import sys
import os
from os.path import join
import cv2 as cv
import numpy as np
sys.path.append(join(os.getcwd(), 'vframe_cli'))
from vframe.settings import app_cfg

# %% codecell
fp_onnx = '/work/vframe/3rdparty/yolov5/weights/yolov5s.onnx'
# %% codecell
net = cv.dnn.readNetFromONNX(fp_onnx)

# %% codecell
blob = cv.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
net.setInput(blob)
detections = net.forward(ln)

# %% codecell
boxes = []
confidences = []
classIDs = []
for output in detections:
    for detection in output:
        scores = detection[5:]
        classID = np.argmax(scores)
        confidence = scores[classID]
        if confidence > args["confidence"]:
            # W, H are the dimensions of the input image
            box = detection[0:4] * np.array([W, H, W, H])
            (centerX, centerY, width, height) = box.astype("int")
            x = int(centerX - (width / 2))
            y = int(centerY - (height / 2))
            boxes.append([x, y, int(width), int(height)])
            confidences.append(float(confidence))
            classIDs.append(classID)
idxs = cv.dnn.NMSBoxes(boxes, confidences, confidence, threshold)
