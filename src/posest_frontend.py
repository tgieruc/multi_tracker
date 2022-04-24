#!/usr/bin/env python3

import time

import cv2
import numpy as np

import rospy
from cv_bridge import CvBridge
from detector import Detector
from sensor_msgs.msg import Image
from tracker import Tracker
from quad_msgs.msg import AngledBox
from lib.utils import create_angled_box, box8to5

import rospkg


class Node_control(object):
    def __init__(self):

        self.frame = None
        self.new_image = False
        self.bridge = CvBridge()
        self.params = {}
        self.get_param()
        if self.params["visualize"]:
            self.visualizer = Visualizer(self.params["input"])
        self.frontend = Frontend(self.params)

        rospy.Subscriber(self.params["input"], Image, self.image_callback)
        self.pub = rospy.Publisher("/posest/AngledBox", AngledBox)

    def get_param(self):
        detector = rospy.get_param("/posest/frontend/detector")
        detector_weights = rospy.get_param("/posest/frontend/detector_weights")
        detector_config = rospy.get_param("posest/frontend/detector_config")
        tracker_config = rospy.get_param("/posest/frontend/tracker_config")
        tracker_weights = rospy.get_param("/posest/frontend/tracker_weights")
        input = rospy.get_param("/posest/frontend/input")
        visualize = rospy.get_param("/posest/frontend/visualize")
        detector_only = rospy.get_param("/posest/frontend/detector_only")
        redetect_time = rospy.get_param("/posest/frontend/redetect_time")

        self.params = {"detector": detector,
                       "detector_weights": detector_weights,
                       "detector_config": detector_config,
                       "tracker_config": tracker_config,
                       "tracker_weights": tracker_weights,
                       "input": input,
                       "visualize": visualize,
                       "detector_only": detector_only,
                       "redetect_time": redetect_time
                       }

    def spin(self):
        self.wait_for_new_image()
        detected, box, mask = self.frontend.detect(self.frame)
        if self.params["visualize"]:
            self.visualizer.show(self.frame, detected, box, mask)
        if detected:
            self.pub.publish(AngledBox(self.box8to5(box).tolist()))


    def wait_for_new_image(self):
        """ waits until image_callback announce a new image"""
        while not self.new_image:
            pass
        self.new_image = False

    def image_callback(self, ros_image):
        """Callback when a new image arrives, transforms it in cv2 image and set self.new_image to True"""
        frame = self.bridge.imgmsg_to_cv2(ros_image, "bgr8")

        self.frame = np.array(frame, dtype=np.uint8)
        self.new_image = True


class Visualizer(object):
    def __init__(self, input_choice):
        self.input = input_choice
        cv2.namedWindow(input_choice, cv2.WND_PROP_FULLSCREEN)

    def show(self, frame, detected, box, mask):
        """Display the frame in a cv2 window. Box is either in xyxy format or polygon xyxyxyxy"""
        if box is not None and detected:
            cv2.polylines(frame, [box.reshape((-1, 1, 2))],
                          True, (0, 255, 0), 3)
            if mask is not None:
                frame = cv2.addWeighted(frame, 0.77, mask, 0.23, -1)
        cv2.imshow(self.input, frame)
        cv2.waitKey(3)


class Frontend(object):
    def __init__(self, params):
        self.params = params
        rospack = rospkg.RosPack()
        path = rospack.get_path('posest_frontend')
        self.detector = Detector(self.params, path)
        self.tracker = Tracker(self.params, path)
        self.detected = False
        self.last_detection_time = time.time()
        self.position = None

    def first_detection(self, frame):
        self.detected, box = self.detector.inference(frame)
        if self.detected:
            self.tracker.init_tracker(frame, box.numpy())
            self.position = np.mean(box.numpy().reshape(2, 2), axis=0)
            return box
        return None

    def check_detection(self, frame):
        self.last_detection_time = time.time()
        self.detected, new_box = self.detector.inference(frame)
        if self.detected:
            new_position = np.mean(new_box.numpy().reshape(2, 2), axis=0)
            if np.linalg.norm(new_position - self.position) > frame.shape[0] / 10:
                print("Tracker reinitialized")
                self.tracker.init_tracker(frame, new_box.numpy())
                self.position = new_position

    def detect(self, frame):
        mask = None

        if self.params["detector_only"]:
            self.detected, box = self.detector.inference(frame)
        else:
            if not self.detected:
                box = self.first_detection(frame)
            else:
                # Check if new initialization of tracker is needed
                if ((time.time() - self.last_detection_time) > self.params["redetect_time"]) and (self.params["redetect_time"] != 0):
                    self.check_detection(frame)

                box, mask = self.tracker.track_frame(frame)

        return self.detected, create_angled_box(box), mask


if __name__ == '__main__':
    rospy.init_node('posest_frontend_node', anonymous=True)
    loop_rate = rospy.Rate(10)

    my_node = Node_control()
    while not rospy.is_shutdown():
        my_node.spin()
        loop_rate.sleep()
