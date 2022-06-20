import sys, os
import torch
import numpy as np
sys.path.insert(0,os.path.join(os.path.dirname(__file__), "lib/pysot"))

from pysot.core.config import cfg
from pysot.models.model_builder import ModelBuilder
from pysot.tracker.tracker_builder import build_tracker




class Tracker(object):
    def __init__(self, params, path):
        cfg.merge_from_file(path + '/modules_config/' + params["tracker_config"])
        cfg.CUDA = torch.cuda.is_available() and cfg.CUDA
        device = torch.device('cuda' if cfg.CUDA else 'cpu')

        # create model
        model = ModelBuilder()

        # load model
        model.load_state_dict(torch.load(path + '/modules_config/' + params["tracker_weights"],
                              map_location=lambda storage, loc: storage.cpu()))
        model.eval().to(device)

        # build tracker
        self.tracker = build_tracker(model)

    def init_tracker(self, frame, init_rect):
        init_rect = init_rect.copy()
        init_rect[2:] -= init_rect[:2]
        self.tracker.init(frame, init_rect)

    def track_frame(self, frame):
        outputs = self.tracker.track(frame)
        if 'polygon' in outputs:
            mask = ((outputs['mask'] > cfg.TRACK.MASK_THERSHOLD) * 255)
            mask = mask.astype(np.uint8)
            mask = np.stack([mask, mask * 255, mask]).transpose(1, 2, 0)
            return np.array(outputs['polygon']).astype(np.int32), mask

        return np.array(outputs['bbox']).astype(np.int32), None