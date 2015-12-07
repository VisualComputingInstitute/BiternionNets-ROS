import numpy as np
import cv2
import sys, os
import argparse
import copy

import DeepFried2 as df
from df_extras import Flatten, Biternion, BiternionCriterion
from training_utils import dotrain, dostats, dopred
from lbtoolbox.util import flipany, printnow
from lbtoolbox.thutil import count_params
from lbtoolbox.augmentation import AugmentationPipeline, Cropper

pjoin = os.path.join

def myimread(fname, zeroone=True, resize=None):
  im = cv2.imread(fname, flags=cv2.IMREAD_COLOR)
  if im is None:
    raise ValueError("Couldn't load image " + fname)

  if resize is not None:
    im = cv2.resize(im, resize, interpolation=cv2.INTER_LANCZOS4)

  # In OpenCV, color dimension is last, but theano likes it to be first.
  # (That's map of triplets vs three maps philosophy.)
  im = np.rollaxis(im, 2, 0) #TODO add comment or vars instead of bare numbers
  return im.astype(np.float32)/255 if zeroone else im

def load(path, testname, skip, ydict):
  Xtr, Xte = [], []
  ytr, yte = [], []
  ntr, nte = [], []

  for lbl in os.listdir(path):
    for f in os.listdir(pjoin(path, lbl)):
      if f.startswith(testname):
        Xte.append(myimread(pjoin(path, lbl, f)))
        yte.append(ydict[lbl])
        nte.append(f)
      elif not any(f.startswith(s) for s in skip):
        Xtr.append(myimread(pjoin(path, lbl, f)))
        ytr.append(ydict[lbl])
        ntr.append(f)

  return np.array(Xtr), np.array(Xte), np.array(ytr), np.array(yte), ntr, nte

def merge4to8(X, y, n):
  y8 = np.full_like(y['4p'], np.nan)
  for idx4p, n4p in enumerate(n['4p']):
    idx4x = n['4x'].index(n4p)
    y4x, y4p = y['4x'][idx4x], y['4p'][idx4p]
    y8[idx4p] = y4x + y4p if not (y4x == 0 and y4p == 3) else 7
  return X['4p'].copy(), y8, n['4p'].copy()

def flipped(X, y, n, old, new):
    indices = np.where(y == old)[0]
    return flipany(X[indices], dim=3), np.full(len(indices), new, dtype=y.dtype), [n[i] for i in indices]

def flipall(X, y, n, flips, append=True):
  fx, fy, fn = [], [], []
  for old, new in flips:
    a, b, c = flipped(X, y, n, old, new)
    fx.append(a) ; fy.append(b) ; fn.append(c)
  if append:
    return np.concatenate([X] + fx), np.concatenate([y] + fy), n + sum(fn, list())
  else:
    return np.concatenate(fx), np.concatenate(fy), sum(fn,list())

def mknet(*outlayers):
  return df.Sequential(                          #     3@46
    df.SpatialConvolutionCUDNN( 3, 24, 3, 3),  # -> 24@44
    df.BatchNormalization(24),
    df.ReLU(),
    df.SpatialConvolutionCUDNN(24, 24, 3, 3),  # -> 24@42
    df.BatchNormalization(24),
    df.SpatialMaxPoolingCUDNN(2, 2),           # -> 24@21
    df.ReLU(),
    df.SpatialConvolutionCUDNN(24, 48, 3, 3),  # -> 48@19
    df.BatchNormalization(48),
    df.ReLU(),
    df.SpatialConvolutionCUDNN(48, 48, 3, 3),  # -> 48@17
    df.BatchNormalization(48),
    df.SpatialMaxPooling(2, 2),                # -> 48@9
    df.ReLU(),
    df.SpatialConvolutionCUDNN(48, 64, 3, 3),  # -> 48@7
    df.BatchNormalization(64),
    df.ReLU(),
    df.SpatialConvolutionCUDNN(64, 64, 3, 3),  # -> 48@5
    df.BatchNormalization(64),
    df.ReLU(),
    df.Dropout(0.2),
    Flatten(),
    df.Linear(64*5*5, 512),
    df.ReLU(),
    df.Dropout(0.5),
    *outlayers
  )

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

def dopred_deg(model, aug, X, batchsize=100):
  return np.rad2deg(dopred(model, aug, X, ensembling=ensemble_degrees, output2preds=lambda x: x, batchsize=batchsize))

def maad_from_deg(preds, reals):
  return np.rad2deg(np.abs(np.arctan2(np.sin(np.deg2rad(reals-preds)), np.cos(np.deg2rad(reals-preds)))))

def prepare_data(datadir):
  classes4x = ['front','right','back','left']
  classnums4x = {c: i for i, c in enumerate(classes4x)}
  classes4p = ['frontright','backright','backleft','frontleft']
  classnums4p = {c: i for i, c in enumerate(classes4p)}
  classes8 = ['frontright','rightfront','rightback','backright','backleft','leftback','leftfront','frontleft']

  centre8_deg = {
      'frontright': 22.5,
      'rightfront': 67.5,
      'rightback': 112.5,
      'backright': 157.5,
      'backleft': 202.5,
      'leftback': 247.5,
      'leftfront': 292.5,
      'frontleft': 337.5,
  }
  centre8_vec = {k: deg2bit(v) for k, v in centre8_deg.items()}

  #f stands for 'flipped'
  Xtr, Xte, Xte_f = {}, {}, {}
  ytr, yte, yte_f = {}, {}, {}
  ntr, nte, nte_f = {}, {}, {}

  for name, ydict in {'4x': classnums4x, '4p': classnums4p}.items():
    Xtr[name], Xte[name], ytr[name], yte[name], ntr[name], nte[name] = load(pjoin(datadir, name),
      testname='lucas', skip=['.', 'dog', 'dog2', 'doggy'], ydict=ydict
    )
  for name in Xtr:
    print(name)
    print("Trainset: X({}), y({})".format(Xtr[name].shape, ytr[name].shape))
    print("Testset: X({}), y({})".format(Xte[name].shape, yte[name].shape))
    print("Random label: {}".format(set(ytr[name])))
  # Do flip-augmentation beforehand.

  Xtr['4x'], ytr['4x'], ntr['4x'] = flipall(Xtr['4x'], ytr['4x'], ntr['4x'], flips=[
      (classnums4x['front'], classnums4x['front']),
      (classnums4x['back'], classnums4x['back']),
      (classnums4x['left'], classnums4x['right']),
      (classnums4x['right'], classnums4x['left']),
  ])

  Xtr['4p'], ytr['4p'], ntr['4p'] = flipall(Xtr['4p'], ytr['4p'], ntr['4p'], flips=[
      (classnums4p['frontleft'], classnums4p['frontright']),
      (classnums4p['frontright'], classnums4p['frontleft']),
      (classnums4p['backleft'], classnums4p['backright']),
      (classnums4p['backright'], classnums4p['backleft']),
  ])
   
  Xte_f['4x'], yte_f['4x'], nte_f['4x'] = flipall(Xte['4x'], yte['4x'], nte['4x'], flips=[
      (classnums4x['front'], classnums4x['front']),
      (classnums4x['back'], classnums4x['back']),
      (classnums4x['left'], classnums4x['right']),
      (classnums4x['right'], classnums4x['left'])],
      append=False
    )
  Xte_f['4p'], yte_f['4p'], nte_f['4p'] = flipall(Xte['4p'], yte['4p'], nte['4p'], flips=[
      (classnums4p['frontleft'], classnums4p['frontright']),
      (classnums4p['frontright'], classnums4p['frontleft']),
      (classnums4p['backleft'], classnums4p['backright']),
      (classnums4p['backright'], classnums4p['backleft'])],
      append=False
    )
  
  
  # Merge 4x and 4p into 8
  Xtr['8'], ytr['8'], ntr['8'] = merge4to8(Xtr, ytr, ntr)
  Xte['8'], yte['8'], nte['8'] = merge4to8(Xte, yte, nte)
  Xte_f['8'], yte_f['8'], nte_f['8'] = merge4to8(Xte_f, yte_f, nte)

  # Convert class-IDs into biternions.
  ytr = np.array([centre8_vec[classes8[y]]for y in ytr['8']])
  yte = np.array([centre8_vec[classes8[y]]for y in yte['8']])
  yte_f = np.array([centre8_vec[classes8[y]]for y in yte_f['8']])
  return Xtr['8'], ytr, Xte['8'], yte, Xte_f['8'], yte_f, nte['8'], nte_f['8']

if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='BiternionNet training')
  parser.add_argument("-c", "--criterion",
    type=str, default='cosine',
    help='Training criterion: `cosine` or `von-mises`',
  )
  parser.add_argument("-e", "--epochs",
    type=int, default=3,
    help='Number of epochs to train.'
  )
  parser.add_argument("-d", "--datadir",
    type=str, default=".",
    help="Location of training data. Needs `4x` and `4p` subfolders."
  )
  parser.add_argument("-o", "--output",
    type=argparse.FileType('w'), default="biternion-net.npz",
    help="File to save the learned model as."
  )

  args = parser.parse_args()
  print(args.criterion + " criterion will be used")

  if args.criterion == 'cosine':
    crit = BiternionCriterion()
  elif args.criterion == 'von-mises':
    crit = BiternionCriterion(kappa=1)
  else:
    print("ERROR: You specified wrong criterion. Sorry =(")
    sys.exit(1)

  printnow("Loading data from {}", args.datadir)
  Xtr, ytr, Xte, yte, Xte_f, yte_f, nte, nte_f = prepare_data(args.datadir)
  ytr = ytr.astype(df.floatX)
  yte = yte.astype(df.floatX)
  yte_f = yte_f.astype(df.floatX)
  printnow("Got {:.2f}k training images after flipping", len(Xtr)/1000)

  aug = AugmentationPipeline(Xtr, ytr, Cropper((46,46)))
  net = mknet(df.Linear(512, 2, initW=df.init.normal(0.01)), Biternion())
  printnow('Network has {:.3f}M params in {} layers', count_params(net)/1000000, len(net.modules))

  costs = dotrain(net, crit, aug, Xtr, ytr, nepochs=args.epochs)
  print("Costs: {}".format(' ; '.join(map(str, costs))))

  dostats(net, aug, Xtr, batchsize=1000)

  # Save the network.
  #printnow("Saving the learned network to {}", args.output)
  #np.savez_compressed(args.output, net.__getstate__())

  # Prediction, TODO: Move to ROS node.
  s = np.argsort(nte)
  Xte,yte,Xte_f,yte_f = Xte[s],yte[s],Xte_f[s],yte_f[s]
  
  printnow("(TEMP) Doing predictions.", args.output)
  y_pred = bit2deg(dopred_deg(net, aug, Xte))
  
  y_pred_f = dopred_deg(net, aug, Xte_f)
  y_pred_f[:,1]=-y_pred_f[:,1]
  y_pred_f = bit2deg(y_pred_f)
  
  res = maad_from_deg(y_pred, bit2deg(yte))
  res2 = maad_from_deg(np.rad2deg(ensemble_degrees([y_pred,y_pred_f])),bit2deg(yte))
  print("Finished predictions")
  print("Mean angle error for train images             = ", res.mean())
  print("Mean angle error for flipped augmented images = ", res2.mean())
