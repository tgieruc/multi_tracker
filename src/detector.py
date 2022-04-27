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
                                        path + '/frontend_config/' + self.params["detector_weights"], source='local',
                                        force_reload=True)

        elif self.params["detector"] == "Detectron2":
            cfg = get_cfg()
            cfg.merge_from_file(path + '/frontend_config/' + self.params["detector_config"])
            # cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
            # cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.4

            cfg.MODEL.WEIGHTS = path + '/frontend_config/' + self.params["detector_weights"]

            self.model = DefaultPredictor(cfg)

        else:
            sys.exit('Select either YOLOv5 or Detectron2')

        print("--.Model loaded.--")

    def inference(self, img):
        results = self.model(img)
        if self.params["detector"] == "YOLOv5":
            if len(results.xyxy[0]) == 0:
                return 0, None
            else:
                index_threshold = results.xyxy[0].data[:, 4] > 0.5
                index_threshold[self.params["number_drones"]-1:] = False
                return len(torch.where(index_threshold == True)), results.xyxy[0].data[index_threshold,:4].int().cpu().numpy()
        else:
            if len(results) > 0:
                if len(results["instances"].pred_boxes) > 0:
                    box = results["instances"].pred_boxes.tensor[0].int().cpu()
                    return True, box

        return False, None
