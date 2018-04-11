import numpy as np


def normalized(x, axis=-1):
    return x / (np.linalg.norm(x, axis=axis, keepdims=True) + 1e-12)


def sigmoid(x):
    return np.exp(-np.logaddexp(0, -x))


def deg2bit(deg):
  rad = np.deg2rad(deg)
  return np.array([np.cos(rad), np.sin(rad)]).T


def bit2deg(bit):
  return (np.rad2deg(np.arctan2(bit[...,1], bit[...,0])) + 360) % 360


def flipbiternions(bits):
  bits = bits.copy()
  bits[...,1] *= -1
  return bits


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


def subtractbg(rgb, depth, threshold, bgcoeff):
    #rgb.flags.writeable = True #super cool hack
    rgb = rgb.copy()

    nanmask = np.isnan(depth)

    # Sometimes we get broken depth of all-nans.
    # We don't want to get an exception from percentile functions then.
    if np.all(nanmask):
        return rgb

    try:
        med = np.nanpercentile(depth, bgcoeff*100)
    except AttributeError:
        # `nanpercentile` only exists starting with numpy 1.9, but Ubuntu 14.04 has 1.8
        med = np.percentile(depth[~nanmask], bgcoeff*100)

    rgb[nanmask] = [0,0,0]
    rgb[med+threshold < depth] = [0,0,0]
    return rgb


def cutout(img, x, y, w, h):
    # Need to be careful for negative indices in conjunction with
    # numpy's (and thus OpenCV's) wrap-around.
    y2, x2 = y+h, x+w
    y1, x1 = max(y, 0), max(x, 0)
    return img[y1:y2, x1:x2]


def cutout_hwfact(x,y,w,h, hfact, wfact):
    # NOTE: Order is important here.
    h = int(round(min(hfact*w, h) if hfact > 0 else h))
    x = x + int(round((1 - wfact)/2*w))
    w = int(round(wfact*w))
    return x, y, w, h
