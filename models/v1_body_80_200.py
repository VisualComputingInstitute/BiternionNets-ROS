import cv2
import numpy as np
import DeepFried2 as df
import common as C
from lbtoolbox.augmentation import AugmentationPipeline, Cropper
from df_extras import Flatten, Biternion

from models.v1 import ModelV1

class Model(ModelV1):
    def __init__(self, weightsname, *unused, **unused_kw):
        self._net = df.Sequential(                  # 184x76
            df.SpatialConvolution( 3, 24, (3, 3)),  # 182x74
            df.BatchNormalization(24),
            df.ReLU(),
            df.SpatialConvolution(24, 24, (3, 3)),  # 180x72
            df.SpatialMaxPooling((3, 3)),           #  60x24
            df.BatchNormalization(24),
            df.ReLU(),
            df.SpatialConvolution(24, 48, (3, 3)),  #  58x22
            df.BatchNormalization(48),
            df.ReLU(),
            df.SpatialConvolution(48, 48, (3, 3)),  #  56x20
            df.SpatialMaxPooling((2, 2)),           #  28x10
            df.BatchNormalization(48),
            df.ReLU(),
            df.SpatialConvolution(48, 64, (3, 3)),  #  26x8
            df.BatchNormalization(64),
            df.ReLU(),
            df.SpatialConvolution(64, 64, (3, 3)),  #  24x6
            df.SpatialMaxPooling((2, 2)),           #  12x3
            df.BatchNormalization(64),
            df.ReLU(),
            df.SpatialConvolution(64, 64, (3, 2)),  #  10x2
            df.BatchNormalization(64),
            df.ReLU(),
            df.Dropout(0.2),
            Flatten(),
            df.Linear(64*10*2, 512),
            df.ReLU(),
            df.Dropout(0.5),
            df.Linear(512, 2, init=df.init.normal(0.01)),
            Biternion()
        )

        self._net.__setstate__(np.load(weightsname))
        self._net.evaluate()

        self._aug = AugmentationPipeline(None, None, Cropper((184,76)))

    def getrect(self, x, y, w, h):
        # Here we use the full box.
        # We know from the detector that full-height = 3x width.
        # If that's more than is seen on camera, it will be clipped.
        return x, y+int(w*0.8), w, 2*w

    def _preproc(self, det_rgb, det_d):
        im = C.subtractbg(det_rgb, det_d, 1.0, 0.5)
        im = cv2.resize(im, (80, 200))
        im = np.rollaxis(im, 2, 0)
        return im.astype(df.floatX)/255
