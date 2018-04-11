import cv2
import numpy as np
import DeepFried2 as df
import common as C
from lbtoolbox.augmentation import AugmentationPipeline, Cropper
from df_extras import Flatten, Biternion

from models.v1 import ModelV1

class Model(ModelV1):
    def __init__(self, weightsname, *unused, **unused_kw):
        self._net = df.Sequential(                  #     3@46
            df.SpatialConvolution( 3, 24, (3, 3)),  # -> 24@44
            df.BatchNormalization(24),
            df.ReLU(),
            df.SpatialConvolution(24, 24, (3, 3)),  # -> 24@42
            df.BatchNormalization(24),
            df.SpatialMaxPooling((2, 2), ignore_border=False),  # -> 24@21
            df.ReLU(),
            df.SpatialConvolution(24, 48, (3, 3)),  # -> 48@19
            df.BatchNormalization(48),
            df.ReLU(),
            df.SpatialConvolution(48, 48, (3, 3)),  # -> 48@17
            df.BatchNormalization(48),
            df.SpatialMaxPooling((2, 2), ignore_border=False),  # -> 48@9
            df.ReLU(),
            df.SpatialConvolution(48, 64, (3, 3)),  # -> 64@7
            df.BatchNormalization(64),
            df.ReLU(),
            df.SpatialConvolution(64, 64, (3, 3)),  # -> 64@5
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

        self._aug = AugmentationPipeline(None, None, Cropper((46, 46)))

    def getrect(self, x, y, w, h):
        # Take only the square upper-body section.
        return x, y, w, w

    def _preproc(self, det_rgb, det_d):
        im = C.subtractbg(det_rgb, det_d, 1.0, 0.5)
        im = cv2.resize(im, (50, 50))
        im = np.rollaxis(im, 2, 0)
        return im.astype(df.floatX)/255
