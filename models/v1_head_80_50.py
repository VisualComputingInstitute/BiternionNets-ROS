import cv2
import numpy as np
import DeepFried2 as df
import common as C
from lbtoolbox.augmentation import AugmentationPipeline, Cropper
from df_extras import Flatten, Biternion

from models.v1 import ModelV1

class Model(ModelV1):
    def __init__(self, weightsname, *unused, **unused_kw):
        self._net = df.Sequential(                  # 48x70  (HxW)
            df.SpatialConvolution( 3, 24, (3, 3)),  # 46x68
            df.BatchNormalization(24),
            df.ReLU(),
            df.SpatialConvolution(24, 24, (3, 3)),  # 44x66
            df.BatchNormalization(24),
            df.SpatialMaxPooling((2, 3)),           # 22x22
            df.ReLU(),
            df.SpatialConvolution(24, 48, (3, 3)),  # 20x20
            df.BatchNormalization(48),
            df.ReLU(),
            df.SpatialConvolution(48, 48, (3, 3)),  # 18x18
            df.BatchNormalization(48),
            df.SpatialMaxPooling((2, 2)),           # 9x9
            df.ReLU(),
            df.SpatialConvolution(48, 64, (3, 3)),  # 7x7
            df.BatchNormalization(64),
            df.ReLU(),
            df.SpatialConvolution(64, 64, (3, 3)),  # 5x5
            df.BatchNormalization(64),
            df.ReLU(),
            df.Dropout(0.2),
            Flatten(),
            df.Linear(64*5*5, 512),
            df.ReLU(),
            df.Dropout(0.5),
            df.Linear(512, 2, init=df.init.normal(0.01)),
            Biternion()
        )

        self._net.__setstate__(np.load(weightsname))
        self._net.evaluate()

        self._aug = AugmentationPipeline(None, None, Cropper((48,70)))

    def getrect(self, x, y, w, h):
        # Take only the square upper-body section.
        # return x, y, int(w*0.8), int(w*0.5)
        return int(x+.1*w), y, int(w*0.8), int(w*0.5)

    def _preproc(self, det_rgb, det_d):
        im = C.subtractbg(det_rgb, det_d, 1.0, 0.5)
        im = cv2.resize(im, (80, 54))
        im = np.rollaxis(im, 2, 0)
        return im.astype(df.floatX)/255
