import cv2
import numpy as np
import DeepFried2 as df
from lbtoolbox.augmentation import AugmentationPipeline, Cropper
from df_extras import Flatten, Biternion


def mknet():
  return df.Sequential(                     #     3@46
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


def mkaug(Xtr, ytr):
  return AugmentationPipeline(Xtr, ytr, Cropper((46,46)))

def preproc(im):
  im = cv2.resize(im, (50, 50))
  im = np.rollaxis(im, 2, 0)
  return im.astype(df.floatX)/255

def cutout(x,y,w,h):
  # Take only the square upper-body section.
  return x,y,w,w
