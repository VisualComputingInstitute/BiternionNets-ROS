import numpy as np
import DeepFried2 as df
from df_extras import Flatten, Biternion


def mknet():
  return df.Sequential(                   #     3@46
    df.SpatialConvolution( 3, 24, 3, 3),  # -> 24@44
    df.BatchNormalization(24),
    df.ReLU(),
    df.SpatialConvolution(24, 24, 3, 3),  # -> 24@42
    df.BatchNormalization(24),
    df.SpatialMaxPooling(2, 2),           # -> 24@21
    df.ReLU(),
    df.SpatialConvolution(24, 48, 3, 3),  # -> 48@19
    df.BatchNormalization(48),
    df.ReLU(),
    df.SpatialConvolution(48, 48, 3, 3),  # -> 48@17
    df.BatchNormalization(48),
    df.SpatialMaxPooling(2, 2),           # -> 48@9
    df.ReLU(),
    df.SpatialConvolution(48, 64, 3, 3),  # -> 48@7
    df.BatchNormalization(64),
    df.ReLU(),
    df.SpatialConvolution(64, 64, 3, 3),  # -> 48@5
    df.BatchNormalization(64),
    df.ReLU(),
    df.Dropout(0.2),
    Flatten(),
    df.Linear(64*5*5, 512),
    df.ReLU(),
    df.Dropout(0.5),
    df.Linear(512, 2, initW=df.init.normal(0.01)),
    Biternion()
  )


def deg2bit(deg):
  rad = np.deg2rad(deg)
  return np.array([np.cos(rad), np.sin(rad)]).T


def bit2deg(angles_bit):
  return (np.rad2deg(np.arctan2(angles_bit[:,1], angles_bit[:,0])) + 360) % 360


def ensemble_degrees(angles):
  return np.arctan2(np.mean(np.sin(np.deg2rad(angles)), axis=0), np.mean(np.cos(np.deg2rad(angles)), axis=0))
