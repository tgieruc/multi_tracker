import time
import rospkg
import numpy as np

from detector import Detector
from tracker import Tracker
from lib.utils import create_angled_box


class Frontend(object):
    def __init__(self, params):
        self.params = params
        rospack = rospkg.RosPack()
        path = rospack.get_path('posest_frontend')
        self.detector = Detector(self.params, path)
        self.bbox_identifier = BboxIdentifier(100, self.params["number_drones"] - 1)

        self.tracker = []
        for i in range(self.params["number_drones"] - 1):
            self.tracker.append(Tracker(self.params, path))
        self.active_tracker = np.zeros(self.params["number_drones"] - 1, dtype=bool)
        self.id_to_tracker = {}
        self.n_detection = 0
        self.last_detection_time = time.time()
        self.object_position = []

    def first_detection(self, frame):
        self.n_detection, boxes = self.detector.inference(frame)
        if self.n_detection > 0:
            self.bbox_identifier.update(boxes)
            self.add_tracker(self.bbox_identifier.id, frame, boxes)

        return boxes

    def add_tracker(self, id_to_add, frame, new_boxes):
        if id_to_add is not None:
            for id in id_to_add:
                free_tracker = np.argwhere(self.active_tracker == False).flatten()
                if len(free_tracker) > 0:
                    new_tracker_id = free_tracker[0]
                    self.id_to_tracker[id] = new_tracker_id
                    self.active_tracker[new_tracker_id] = True
                    self.tracker[new_tracker_id].init_tracker(frame, new_boxes[new_tracker_id])

    def remove_tracker(self, id_to_remove):
        for id in id_to_remove:
            self.active_tracker[self.id_to_tracker[id]] = False

    def check_detection(self, frame):
        self.last_detection_time = time.time()
        self.n_detection, new_boxes = self.detector.inference(frame)
        prev_id = self.bbox_identifier.id
        self.bbox_identifier.update(new_boxes)
        new_id = self.bbox_identifier.id

        for id in prev_id:
            if (new_id ==id).any():
                new_id = np.delete(new_id, np.argwhere(new_id==id))
                prev_id = np.delete(prev_id, np.argwhere(prev_id==id))
        self.add_tracker(new_id, frame, new_boxes)
        self.remove_tracker(prev_id)


        # if (self.bbox_identifier.id is None) and (prev_id is not None):
        #     self.active_tracker = np.zeros(self.params["number_drones"] - 1, dtype=bool)
        # elif len(self.bbox_identifier.id) > len(prev_id):
        #     free_tracker = np.where(self.active_tracker == False)
        #     if len(free_tracker[0]) > 0:
        #         for id in self.bbox_identifier.id:
        #             if not (prev_id == id).any():
        #                 new_tracker_id = free_tracker[0][0]
        #                 self.id_to_tracker[id] = new_tracker_id
        #                 self.active_tracker[new_tracker_id] = True
        #                 self.tracker[new_tracker_id].init_tracker(frame, new_boxes[new_tracker_id])
        # elif len(self.bbox_identifier.id) < len(prev_id):
        #     for i in prev_id:
        #         if not (self.bbox_identifier.id == i).any():
        #             self.active_tracker[self.id_to_tracker[i]] = False

    def detect(self, frame):
        masks = None

        if self.params["detector_only"]:
            self.n_detection, boxes = self.detector.inference(frame)
        else:
            if self.n_detection == 0:
                boxes = self.first_detection(frame)
            else:
                # Check if new initialization of tracker is needed
                if ((time.time() - self.last_detection_time) > self.params["redetect_time"]) and (
                        self.params["redetect_time"] != 0):
                    self.check_detection(frame)

                boxes = []
                masks = []
                for id in np.argwhere(self.active_tracker == True).flatten():
                    box, mask = self.tracker[id].track_frame(frame)
                    boxes.append(box)
                    masks.append(mask)
                if len(boxes) > 0:
                    boxes = np.array(boxes)
                else:
                    boxes = None
                if len(masks) > 0:
                    masks = np.array(masks).max(0)
                else:
                    masks = None

        return self.n_detection, create_angled_box(boxes), masks


class BboxIdentifier(object):
    def __init__(self, threshold, max_id):
        self.old_predictions = None
        self.center = None
        self.id = None
        self.threshold = threshold
        self.max_id = max_id

    def update(self, new_prediction):
        # if nothing n_detection, reset
        if new_prediction is None:
            self.old_predictions = None
            self.center = None
            self.id = None
        elif self.old_predictions is None:
            # if no predictions was done, initialize
            self._initialize(new_prediction)
        else:
            self._update_predictions(new_prediction)

    def _initialize(self, new_prediction):
        self.old_predictions = new_prediction
        self.id = np.arange(len(new_prediction))
        self.center = new_prediction.reshape(-1, 2, 2).mean(axis=1)

    def _update_predictions(self, new_prediction):
        new_center = np.zeros((len(new_prediction), 2))
        new_id = np.arange(max(self.id) + 1, max(self.id) + 1 + len(new_prediction))
        for i in range(len(new_prediction)):
            new_center[i] = new_prediction[i].reshape(-1, 2).mean(axis=0)
            threshold = (new_prediction.reshape(-1, 2, 2)[i, 1, :] - new_prediction.reshape(-1, 2, 2)[i, 0, :])
            for idx, center in enumerate(self.center):
                if (np.linalg.norm(new_center[i] - center) < 2*threshold).all():
                    new_id[i] = self.id[idx]

        filter = []
        if len(new_center) > 1:
            for i in range(len(new_center) - 1):
                for j in range(i+1, len(new_center)):
                    if np.linalg.norm(new_center[i] - new_center[j]) < 100:
                            filter.append(i)

        self.id = np.delete(new_id, filter)
        self.center = np.delete(new_center, filter, axis=0)