import numpy as np


def deg2bit(deg):
  rad = np.deg2rad(deg)
  return np.array([np.cos(rad), np.sin(rad)]).T


def bit2deg(angles_bit):
  return (np.rad2deg(np.arctan2(angles_bit[:,1], angles_bit[:,0])) + 360) % 360


def ensemble_degrees(angles):
  """
  Averages the `angles` (in degrees) along the first dimension, i.e. an input of
  shape (3, 5) will result in 5 average angle outputs.
  Also works for 1D inputs, where it just computes the average of all.
  NOTE: returned angle is in radians (for now).
  """
  return np.arctan2(np.mean(np.sin(np.deg2rad(angles)), axis=0), np.mean(np.cos(np.deg2rad(angles)), axis=0))


def ensemble_biternions(biternions):
    """
    That's actually easy: it's just vectors, so we can just average!
    """
    return np.mean(biternions, axis=0)
