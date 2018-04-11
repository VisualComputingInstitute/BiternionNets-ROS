import cv2
import numpy as np
import DeepFried2 as df
import common as C
from lbtoolbox.augmentation import AugmentationPipeline, Cropper
from df_extras import Flatten, Biternion

from models.v1 import ModelV1

class Model(ModelV1):
    def __init__(self, weightsname, *unused, **unused_kw):
        self._net = df.Sequential(                  # 220x76
            df.SpatialConvolution( 3, 24, (3, 3)),  # 218x74
            df.BatchNormalization(24),
            df.ReLU(),
            df.SpatialConvolution(24, 24, (3, 3)),  # 216x72
            df.SpatialMaxPooling((3, 3)),           #  72x24
            df.BatchNormalization(24),
            df.ReLU(),
            df.SpatialConvolution(24, 48, (3, 3)),  #  70x22
            df.BatchNormalization(48),
            df.ReLU(),
            df.SpatialConvolution(48, 48, (3, 3)),  #  68x20
            df.SpatialMaxPooling((2, 2)),           #  34x10
            df.BatchNormalization(48),
            df.ReLU(),
            df.SpatialConvolution(48, 64, (3, 3)),  #  32x8
            df.BatchNormalization(64),
            df.ReLU(),
            df.SpatialConvolution(64, 64, (3, 3)),  #  30x6
            df.SpatialMaxPooling((2, 2)),           #  15x3
            df.BatchNormalization(64),
            df.ReLU(),
            df.SpatialConvolution(64, 64, (3, 2)),  #  13x2
            df.BatchNormalization(64),
            df.ReLU(),
            df.Dropout(0.2),
            Flatten(),
            df.Linear(64*13*2, 512),
            df.ReLU(),
            df.Dropout(0.5),
            df.Linear(512, 2, init=df.init.normal(0.01)),
            Biternion()
        )

        self._net.__setstate__(np.load(weightsname))
        self._net.evaluate()

        self._aug = AugmentationPipeline(None, None, Cropper((220,76)))

    def getrect(self, x, y, w, h):
        # Here we use the full box.
        # We know from the detector that full-height = 3x width.
        # If that's more than is seen on camera, it will be clipped.
        return x, y, w, 3*w

    def _preproc(self, det_rgb, det_d):
        im = C.subtractbg(det_rgb, det_d, 1.0, 0.5)
        im = cv2.resize(im, (80, 240))
        im = np.rollaxis(im, 2, 0)
        return im.astype(df.floatX)/255
