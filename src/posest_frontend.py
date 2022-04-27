#!/usr/bin/env python3

import cv2
import numpy as np

import rospy
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from quad_msgs.msg import AngledBox, AngledBoxArray
from lib.utils import box8to5
from frontend import Frontend, Bbox4, Bbox5, Bbox8


class NodeControl(object):
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
        self.pub = rospy.Publisher("/posest/AngledBoxArray", AngledBoxArray, queue_size=10)

    def get_param(self):
        detector = rospy.get_param("/posest/frontend/detector/name")
        detector_weights = rospy.get_param("/posest/frontend/detector/weights")
        detector_config = rospy.get_param("posest/frontend/detector/config")
        tracker_config = rospy.get_param("/posest/frontend/tracker/config")
        tracker_weights = rospy.get_param("/posest/frontend/tracker/weights")
        input = rospy.get_param("/posest/frontend/input")
        visualize = rospy.get_param("/posest/frontend/visualize")
        detector_only = rospy.get_param("/posest/frontend/detector_only")
        redetect_time = rospy.get_param("/posest/frontend/redetect_time")
        number_drones = rospy.get_param("/posest/frontend/number_drones")

        track_thresh = rospy.get_param("/posest/frontent/bytetrack/track_thresh")
        device = rospy.get_param("/posest/frontent/bytetrack/device")
        track_buffer = rospy.get_param("/posest/frontent/bytetrack/track_buffer")
        match_thresh = rospy.get_param("/posest/frontent/bytetrack/match_thresh")

        self.params = {"detector": detector,
                       "detector_weights": detector_weights,
                       "detector_config": detector_config,
                       "tracker_config": tracker_config,
                       "tracker_weights": tracker_weights,
                       "input": input,
                       "visualize": visualize,
                       "detector_only": detector_only,
                       "redetect_time": redetect_time,
                       "number_drones": number_drones,
                       "track_thresh": track_thresh,
                       "device": device,
                       "track_buffer": track_buffer,
                       "match_thresh": match_thresh,
                       }

    def spin(self):
        self.wait_for_new_image()
        detected, boxes, mask = self.frontend.detect(self.frame)

        if self.params["visualize"]:
            self.visualizer.show(self.frame, detected, boxes, mask)

        if boxes is not None:
            angled_box_array = AngledBoxArray()
            angled_box_array.angledbox_array = []
            boxes = boxes.to_bbox5()
            for i, box in enumerate(boxes.bbox):
                angled_box_array.angledbox_array.append(AngledBox(boxes.id[i], box.tolist()))
            self.pub.publish(angled_box_array)

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

    def show(self, frame, detected, boxes, masks):
        """Display the frame in a cv2 window. Box is either in xyxy format or polygon xyxyxyxy"""
        frame = np.ascontiguousarray(np.copy(frame))

        def get_color(idx):
            idx = idx * 3
            color = ((37 * idx) % 255, (17 * idx) % 255, (29 * idx) % 255)

            return color

        if detected > 0:
            boxes = boxes.to_bbox8()

            for i, box in enumerate(boxes.bbox):
                intbox = box.reshape((-1, 1, 2)).astype(int)
                obj_id = int(boxes.id[i])
                id_text = '{}'.format(int(obj_id))
                color = get_color(abs(obj_id))
                cv2.polylines(frame, [intbox], True, color, 3)

                cv2.putText(frame, id_text, (intbox[0, 0, 0], intbox[0, 0, 1]), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255),
                            thickness=2)
            if masks is not None:
                frame = cv2.addWeighted(frame, 0.77, masks, 0.23, -1)
        cv2.imshow(self.input, frame)
        cv2.waitKey(3)


if __name__ == '__main__':
    rospy.init_node('posest_frontend_node', anonymous=True)
    loop_rate = rospy.Rate(10)

    my_node = NodeControl()
    while not rospy.is_shutdown():
        my_node.spin()
        loop_rate.sleep()
