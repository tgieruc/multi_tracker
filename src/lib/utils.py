import torch
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

class Bbox4(object):
    def __init__(self, id, bbox, bbox_type="xyxy"):
        self.id = id
        self.bbox = np.array(bbox).reshape(-1, 4)
        if bbox_type == "xywh":
            self.bbox = np.hstack([self.bbox[:, :2], self.bbox[:, 2:] + self.bbox[:, :2]])

    def to_bbox4(self):
        return self

    def to_bbox5(self):
        box = self.bbox.reshape(-1, 2, 2)
        center = box.mean(2)
        width = 2 * (box.mean(1) - box.reshape(-1, 4)[:, :2])
        return Bbox5(self.id, np.hstack([center, width, np.zeros((box.shape[0], 1))]))

    def to_bbox8(self):
        boxes = self.bbox
        return Bbox8(self.id, np.dstack(
            [boxes[:, 0], boxes[:, 1], boxes[:, 0], boxes[:, 3], boxes[:, 2], boxes[:, 3], boxes[:, 2],
             boxes[:, 1]]).reshape(-1, 8).astype(np.int32))

    def to_torch(self):
        width = 2 * (self.bbox.reshape(-1, 2, 2).mean(1) - self.bbox.reshape(-1, 4)[:, :2])
        return torch.tensor(np.hstack([self.bbox[:,:2], width, 1 * torch.ones((len(self.bbox),1))]))

    def is_inside(self, center):
        boxes = self.bbox.reshape(-1,2,2)
        bbox_has_center = []
        for box in boxes:
            c1 = (center > box[0]).all(1)
            c2 = (center < box[1]).all(1)
            bbox_has_center.append(c1 & c2)
        return np.array(bbox_has_center, dtype=bool)

    def center(self):
        return self.bbox.reshape(-1,2,2).mean(1)



class Bbox5(object):
    def __init__(self, id, bbox):
        self.id = id
        self.bbox = bbox

    def to_bbox4(self):
        return Bbox4(self.id, self.bbox[:, :4])


class Bbox8(object):
    def __init__(self, id, bbox):
        self.id = id
        self.bbox = np.array(bbox).reshape(-1, 8)

    def center(self):
        box = self.bbox.reshape(-1, 4, 2)
        return box.mean(axis=1)

    def to_bbox4(self):
        bbox_reshaped = self.bbox.reshape(-1, 8)
        bbox4 = np.stack([bbox_reshaped[:, 0], bbox_reshaped[:, 1], bbox_reshaped[:, 4], bbox_reshaped[:, 5]]).T
        return Bbox4(self.id, bbox4)

    def to_bbox5(self):
        box = self.bbox.reshape(-1, 4, 2)
        center = box.mean(axis=1)
        box_size = abs(center - box[:, 0])
        ba = box[:, 1] - box[:, 0]
        bc = box[:, 3] - box[:, 0]

        cosine_angle = np.sum(ba * bc, axis=1) / (np.linalg.norm(ba) * np.linalg.norm(bc))
        angle = np.arccos(cosine_angle)
        return Bbox5(self.id, np.dstack([center[:, 0], center[:, 1], box_size[:, 0], box_size[:, 1], angle - np.pi / 2]).reshape(-1, 5))

    def to_bbox8(self):
        return self

