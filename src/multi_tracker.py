#!/usr/bin/env python3

import cv2
import time
import numpy as np

import rospy
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from std_msgs.msg import Float32
from multi_tracker_msgs.msg import AngledBox, AngledBoxArray, MaskArray
from frontend import Frontend


class NodeControl(object):
    def __init__(self):

        self.frame = None
        self.new_image = False
        self.bridge = CvBridge()
        self.params = {}
        self.get_param()
        self.frontend = Frontend(self.params)
        if self.params["visualize"]:
            self.visualizer = Visualizer(self.params["input"])
        rospy.Subscriber(self.params["input"], Image, self.image_callback, queue_size=1000)
        self.bbox_pub = rospy.Publisher("multi_tracker/angledbox_array", AngledBoxArray, queue_size=0)
        self.mask_pub = rospy.Publisher("multi_tracker/mask", MaskArray, queue_size=0)
        self.time_pub = rospy.Publisher("multi_tracker/inference_time", Float32, queue_size=0)


    def get_param(self):
        detector = rospy.get_param("multi_tracker/detector/name")
        detector_weights = rospy.get_param("multi_tracker/detector/weights")
        detector_config = rospy.get_param("multi_tracker/detector/config")
        tracker_config = rospy.get_param("multi_tracker/tracker/config")
        tracker_weights = rospy.get_param("multi_tracker/tracker/weights")
        input = rospy.get_param("multi_tracker/input")
        visualize = rospy.get_param("multi_tracker/visualize")
        tracker = rospy.get_param("multi_tracker/tracker/name")
        redetect_time = rospy.get_param("multi_tracker/redetect_time")
        number_objects = rospy.get_param("multi_tracker/number_object")

        track_thresh = rospy.get_param("multi_tracker/bytetrack/track_thresh")
        device = rospy.get_param("multi_tracker/bytetrack/device")
        track_buffer = rospy.get_param("multi_tracker/bytetrack/track_buffer")
        match_thresh = rospy.get_param("multi_tracker/bytetrack/match_thresh")

        assert (tracker == "PySOT" or tracker == "ByteTrack"), "multi_tracker/tracker/name has to be either ByteTrack or PySOT"
        assert (detector == "YOLOv5" or detector == "Detectron2"), "multi_tracker/tracker/name has to be either YOLOv5 or Detectron2"

        self.params = {"detector": detector,
                       "detector_weights": detector_weights,
                       "detector_config": detector_config,
                       "tracker_config": tracker_config,
                       "tracker_weights": tracker_weights,
                       "input": input,
                       "visualize": visualize,
                       "tracker": tracker,
                       "redetect_time": redetect_time,
                       "number_objects": number_objects,
                       "track_thresh": track_thresh,
                       "device": device,
                       "track_buffer": track_buffer,
                       "match_thresh": match_thresh,
                       }

    def spin(self):
        t0 = time.time()
        rospy.wait_for_message(self.params["input"], Image)
        frame_saved = self.frame
        detected, boxes, mask = self.frontend.detect(self.frame)
        if self.params["visualize"]:
            boxes_save = boxes

        if boxes is not None:
            angled_box_array = AngledBoxArray()
            angled_box_array.angledbox_array = []
            boxes = boxes.to_bbox5()
            for i, box in enumerate(boxes.bbox):
                angled_box_array.angledbox_array.append(AngledBox(boxes.id[i], box.tolist()))
            self.bbox_pub.publish(angled_box_array)
        if mask is not None:
            mask_array = MaskArray()
            for mask_ in mask:
                mask_array.mask_array.append(self.bridge.cv2_to_imgmsg(np.array(mask_)))
            self.mask_pub.publish(mask_array)
        self.time_pub.publish(time.time() - t0)
        if self.params["visualize"]:
            self.visualizer.show(frame_saved, detected, boxes_save, mask, 1 / float(time.time() - t0))

    def image_callback(self, ros_image):
        """Callback when a new image arrives, transforms it in cv2 image and set self.new_image to True"""
        frame = self.bridge.imgmsg_to_cv2(ros_image, "bgr8")

        self.frame = np.array(frame, dtype=np.uint8)
        self.new_image = True


class Visualizer(object):
    def __init__(self, input_choice):
        self.input = input_choice
        cv2.namedWindow(input_choice, cv2.WND_PROP_FULLSCREEN)

    def show(self, frame, detected, boxes, masks, fps):
        """Display the frame in a cv2 window. Box is either in xyxy format or polygon xyxyxyxy"""
        frame = np.ascontiguousarray(np.copy(frame))

        def get_color(idx):
            idx = idx * 3
            color = ((37 * idx) % 255, (17 * idx) % 255, (29 * idx) % 255)

            return color

        if boxes is not None:
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
                frame = cv2.addWeighted(frame, 0.77, np.array(masks).max(0), 0.23, -1)
        # frame = cv2.putText(frame, f'FPS: {fps:.2f}', (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA, False)
        cv2.imshow(self.input, frame)
        cv2.waitKey(3)


if __name__ == '__main__':
    rospy.init_node('multi_tracker_node', anonymous=True)
    loop_rate = rospy.Rate(100)

    my_node = NodeControl()
    while not rospy.is_shutdown():
        my_node.spin()
        # loop_rate.sleep()
