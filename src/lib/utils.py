import os
import cv2
from glob import glob
import numpy as np


def box8to5(box):
    """xy xy xy xy to center x, center y, width, height, angle"""
    box = box.reshape(4, 2)
    center = box.mean(axis=0)
    box_size = abs(center - box[0])
    ba = box[1] - box[0]
    bc = box[3] - box[0]

    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(cosine_angle)
    return np.array([center[0], center[1], box_size[0], box_size[1], angle])


def create_angled_box(boxes):
    if boxes is not None:
        if boxes.shape[1] == 4:
            boxes = np.dstack([boxes[:,0], boxes[:,1], boxes[:,0], boxes[:,3], boxes[:,2], boxes[:,3], boxes[:,2], boxes[:,1]]).reshape(-1,8).astype(np.int32)

    return boxes
