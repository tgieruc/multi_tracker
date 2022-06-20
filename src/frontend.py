import time
import rospkg
import numpy as np
from detector import Detector
from tracker import Tracker
from lib.utils import Bbox4, Bbox8

import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "lib/ByteTrack"))
from yolox.tracker.byte_tracker import BYTETracker


class Frontend(object):
    def __init__(self, params):
        self.params = params
        rospack = rospkg.RosPack()
        path = rospack.get_path('multi_tracker')
        self.detector = Detector(self.params, path)
        if params["tracker"] == "ByteTrack":
            self.bytetracker = BYTETracker(self.params)
        else:
            self.tracker = []
            for i in range(self.params["number_objects"] - 1):
                self.tracker.append(Tracker(self.params, path))
            self.active_tracker = np.zeros(self.params["number_objects"] - 1, dtype=bool)
        self.n_detection = 0
        self.last_detection_time = time.time()
        self.tracker_center = np.zeros((self.params["number_objects"] - 1, 2))
        self.frame = None

    def detection(self):
        self.n_detection, inference = self.detector.inference(self.frame)
        bboxes = None
        if self.n_detection > 0:
            W, H = self.frame.shape[:2]
            online_targets = self.bytetracker.update(inference, [W, H], [W, H])
            id = []
            bbox = []
            for target in online_targets:
                id.append(target.track_id)
                bbox.append(target.tlwh)
            bboxes = Bbox4(id, bbox, "xywh")

        return bboxes

    def detection_tracking(self):
        self.n_detection, inference = self.detector.inference(self.frame)
        if self.n_detection > 0:
            bbox = Bbox4(np.arange(self.n_detection), inference[:, :4])
            self.update_trackers(bbox)
            return bbox
        else:
            return None

    def update_trackers(self, bboxes):
        self.last_detection_time = time.time()

        if bboxes is None:
            self.active_tracker = np.zeros(self.params["number_objects"] - 1, dtype=bool)
        else:
            for _ in range(2 * self.params["number_objects"] - 1):
                bboxes_with_tracker = bboxes.is_inside(self.tracker_center)
                tracker_without_bbox = np.where(bboxes_with_tracker.sum(0) == 0)
                for id in tracker_without_bbox:
                    self.active_tracker[id] = False
                    self.tracker_center[id] = np.array([0, 0])
                trackers_per_bbox = bboxes_with_tracker.sum(1)
                ids_too_many_tracker = np.where(trackers_per_bbox > 1)
                for i in ids_too_many_tracker:
                    tracker_to_disable = np.where(bboxes_with_tracker[i, :] == True)[0][1:]
                    self.active_tracker[tracker_to_disable] = False
                    self.tracker_center[tracker_to_disable] = np.array([0, 0])
                    bboxes_with_tracker[tracker_to_disable, i] = False

                bboxes_without_tracker = np.where((bboxes_with_tracker == True).any(1) == False)[0]

                for i in bboxes_without_tracker:
                    self.tracker[i].init_tracker(self.frame, bboxes.bbox[i])
                    self.active_tracker[i] = True

    def detect(self, frame):
        masks = None
        self.frame = frame
        bboxes = None
        if self.params["tracker"] == "ByteTrack":
            bboxes = self.detection()
        else:
            if (self.n_detection == 0) or (
                    ((time.time() - self.last_detection_time) > self.params["redetect_time"]) and (
                    self.params["redetect_time"] != 0)):
                bboxes = self.detection_tracking()
            if self.n_detection > 0:
                boxes = []
                masks = []
                ids = []
                for id in np.argwhere(self.active_tracker == True).flatten():
                    box, mask = self.tracker[id].track_frame(frame)
                    boxes.append(box)
                    masks.append(mask)
                    ids.append(id)
                if len(boxes) > 0:
                    if boxes[0].shape[0] == 4:
                        bboxes = Bbox4(np.array(ids).flatten(), boxes)
                    elif boxes[0].shape[0] == 8:
                        bboxes = Bbox8(np.array(ids).flatten(), boxes)

                else:
                    bboxes = None
                if len(masks) > 0:
                    masks = np.array(masks).max(0)
                else:
                    masks = None

                self.tracker_center[ids] = bboxes.center()

        return self.n_detection, bboxes, masks
