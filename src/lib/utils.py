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


def create_angled_box(box):
    if box is not None:
        if box.shape == (4,):
            box = np.array([box[0], box[1], box[0], box[3], box[2], box[3], box[2], box[1]])
    return box
