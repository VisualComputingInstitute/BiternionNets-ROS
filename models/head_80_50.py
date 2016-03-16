import cv2
import numpy as np
import DeepFried2 as df
from lbtoolbox.augmentation import AugmentationPipeline, Cropper, Gamma
from df_extras import Flatten, Biternion


def mknet():
  return df.Sequential(                     # 48x70  (HxW)
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


def mkaug(Xtr, ytr):
  return AugmentationPipeline(Xtr, ytr,
    Cropper((48,70)),
    # Gamma(),
  )

def preproc(im):
  im = cv2.resize(im, (80, 54))
  im = np.rollaxis(im, 2, 0)
  return im.astype(df.floatX)/255

def getrect(x,y,w,h):
  # Take only the square upper-body section.
  return x,y,int(w*0.8),int(w*0.5)
