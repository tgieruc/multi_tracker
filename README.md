# Multi-tracker

Multi-tracker is a package for multi-object tracking with ROS Noetic. Its modular design allows the user to chose between [YOLOv5](https://github.com/ultralytics/yolov5) and [Detectron2](https://github.com/facebookresearch/detectron2) as object detectors and [ByteTrack](https://github.com/ifzhang/ByteTrack) or [PySOT](https://github.com/STVIR/pysot) as object trackers.

It is designed as frontend for the [MSL-RAPTOR package](https://github.com/tgieruc/msl_raptor) and requires [multi_tracker_msgs](https://github.com/tgieruc/multi_tracker_msgs) to communicate.

It detects and tracks objects from a ROS topic, specified in the _config files_, and outputs the bounding boxes in _/multi_tracker/angledbox_array_, the inference time in _/multi_tracker/inference_time_ as well as the segmentation masks in _/multi_tracker/mask_, if available.

## Installation
In your ROS workspace source folder, clone the repository with its submodules using 


```bash
git clone --recurse-submodules https://github.com/tgieruc/multi_tracker
``` 

Python dependency have to be installed:
```bash
pip install -r requirements.txt
pip install -r src/lib/detectron2/requirements.txt
``` 

then use Catkin to build the package.


## Settings
ROS configuration file is located in the *config* folder. Two example files are provided, [here](https://github.com/tgieruc/multi_tracker/blob/master/config/multi_tracker_detectron.yaml) for Detectron2 and [here](https://github.com/tgieruc/multi_tracker/blob/master/config/multi_tracker_yolo.yaml) for YOLOv5. All configuration files for the modules have to be stored in the *modules_config* folder.

All paths specified in the ROS config file are relative to the *modules_config* folder. No parameters can be left empty, unused parameters can be set at *""*.

* multi_tracker/detector/name: either *YOLOv5* or *Detectron2*
* multi_tracker/detector/weights: path to the detector weights
* multi_tracker/detector/config: only for Detectron2, the path to the config file
* multi_tracker/tracker/name: either *PySOT* or *ByteTrack* 
* multi_tracker/tracker/config: only for PySOT, path to the config file
* multi_tracker/tracker/weights: only for PySOT, path to the config file
* multi_tracker/input: the ROS camera_raw topic
* multi_tracker/visualize: *true* or *false*, whether to visualize the output
* multi_tracker/redetect_time: only for PySOT, laps of seconds between each redetection
* multi_tracker/number_object: only for PySOT, total number of objects, including the one doing the tracking
* multi_tracker/bytetrack/track_thresh: only for ByteTrack, see ByteTrack documentation for specific information
* multi_tracker/bytetrack/device: either *"cpu"* or *"gpu"*
* multi_tracker/bytetrack/track_buffer: only for ByteTrack, see ByteTrack documentation for specific information
* multi_tracker/bytetrack/match_thresh: only for ByteTrack, see ByteTrack documentation for specific information

# Contact

For any questions, please email me at <theo.gieruc@gmail.com>
