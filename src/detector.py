import torch
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "lib/detectron2"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "lib/yoloV5"))

from detectron2.config import get_cfg
from detectron2.engine.defaults import DefaultPredictor


class Detector(object):
    def __init__(self, params, path):
        self.params = params
        self.model = None
        print("--.Loading detection model.--")
        if self.params["detector"] == "YOLOv5":
            self.model = torch.hub.load(path + '/src/lib/yolov5', 'custom',
                                        path + '/multi_tracker_config/' + self.params["detector_weights"], source='local',
                                        force_reload=True)

        elif self.params["detector"] == "Detectron2":
            cfg = get_cfg()
            cfg.merge_from_file(path + '/multi_tracker_config/' + self.params["detector_config"])
            cfg.MODEL.WEIGHTS = path + '/multi_tracker_config/' + self.params["detector_weights"]

            self.model = DefaultPredictor(cfg)

        else:
            sys.exit('Select either YOLOv5 or Detectron2')

        print("--.Model loaded.--")

    def inference(self, img):
        results = self.model(img)
        if self.params["detector"] == "YOLOv5":
            if len(results.xyxy[0]) > 0:
                index_threshold = results.xyxy[0].data[:, 4] > 0.5
                index_threshold[self.params["number_objects"]-1:] = False
                if self.params["detector_only"]:
                    results.xyxy[0].data[index_threshold, 4] = 1
                return len(torch.where(index_threshold == True)[0]), results.xyxy[0].data[index_threshold,:5].cpu()
        else:
            if len(results) > 0:
                n_result = len(results["instances"].pred_boxes)
                if n_result > 0:
                    box = results["instances"].pred_boxes.tensor.cpu()
                    scores = results["instances"].scores.cpu()
                    index_threshold = scores > 0.5
                    index_threshold[self.params["number_objects"] - 1:] = False

                    return len(torch.where(index_threshold == True)[0]), torch.cat([box, scores[:,None]], dim=1)[index_threshold]

        return 0, None
