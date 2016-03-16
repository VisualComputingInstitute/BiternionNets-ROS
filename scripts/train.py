#!/usr/bin/env python
# encoding: utf-8

import numpy as np
import cv2
import sys, os
import argparse
import copy
from importlib import import_module

import DeepFried2 as df
from lbtoolbox.util import flipany, printnow

from training_utils import dotrain, dostats, dopred
from df_extras import BiternionCriterion
from common import deg2bit, bit2deg, flipbiternions, ensemble_biternions, cutout

pjoin = os.path.join

def myimread(fname, netlib):
  im = cv2.imread(fname, flags=cv2.IMREAD_COLOR)
  if im is None:
    raise ValueError("Couldn't load image " + fname)

  im = cutout(im, *netlib.getrect(0, 0, im.shape[1], im.shape[0]))
  return netlib.preproc(im)

def load(path, testname, skip, ydict, netlib):
  Xtr, Xte = [], []
  ytr, yte = [], []
  ntr, nte = [], []

  for lbl in os.listdir(path):
    for f in os.listdir(pjoin(path, lbl)):
      if f.startswith(testname):
        Xte.append(myimread(pjoin(path, lbl, f), netlib))
        yte.append(ydict[lbl])
        nte.append(f)
      elif not any(f.startswith(s) for s in skip):
        Xtr.append(myimread(pjoin(path, lbl, f), netlib))
        ytr.append(ydict[lbl])
        ntr.append(f)

  return np.array(Xtr), np.array(Xte), np.array(ytr), np.array(yte), ntr, nte

def merge4to8(X, y, n):
  y8 = np.full_like(y['4p'], np.nan)
  for idx4p, n4p in enumerate(n['4p']):
    idx4x = n['4x'].index(n4p)
    y4x, y4p = y['4x'][idx4x], y['4p'][idx4p]
    y8[idx4p] = y4x + y4p if not (y4x == 0 and y4p == 3) else 7
  return X['4p'].copy(), y8, copy.deepcopy(n['4p'])

def flipped(X, y, n, old, new):
    indices = np.where(y == old)[0]
    return flipany(X[indices], dim=3), np.full(len(indices), new, dtype=y.dtype), [n[i] for i in indices]

def flipall(X, y, n, flips):
  fx, fy, fn = [], [], []
  for old, new in flips:
    a, b, c = flipped(X, y, n, old, new)
    fx.append(a) ; fy.append(b) ; fn.append(c)
  return np.concatenate([X] + fx), np.concatenate([y] + fy), n + sum(fn, list())

def dopred_bit(model, aug, X, batchsize=100):
    return dopred(model, aug, X, ensembling=ensemble_biternions, output2preds=lambda x: x, batchsize=batchsize)

def maad_from_deg(preds, reals):
  return np.rad2deg(np.abs(np.arctan2(np.sin(np.deg2rad(reals-preds)), np.cos(np.deg2rad(reals-preds)))))

def skip_one_classers(X,y,n):
  X['4p'] = np.array([X['4p'][i] for i,name in enumerate(n['4p']) if name in n['4x']])
  y['4p'] = np.array([y['4p'][i] for i,name in enumerate(n['4p']) if name in n['4x']])
  n['4p'] = [name for name in n['4p'] if name in n['4x']]

  X['4x'] = np.array([X['4x'][i] for i,name in enumerate(n['4x']) if name in n['4p']])
  y['4x'] = np.array([y['4x'][i] for i,name in enumerate(n['4x']) if name in n['4p']])
  n['4x'] = [name for name in n['4x'] if name in n['4p']]

  return X, y, n

def prepare_data(datadir, netlib):
  classes4x = ['front','right','back','left']
  classnums4x = {c: i for i, c in enumerate(classes4x)}
  classes4p = ['frontright','backright','backleft','frontleft']
  classnums4p = {c: i for i, c in enumerate(classes4p)}
  classes8 = ['frontright','rightfront','rightback','backright','backleft','leftback','leftfront','frontleft']
  classnums8 = {c: i for i, c in enumerate(classes8)}

  centre8 = {
      'frontright': 22.5,
      'rightfront': 67.5,
      'rightback': 112.5,
      'backright': 157.5,
      'backleft': 202.5,
      'leftback': 247.5,
      'leftfront': 292.5,
      'frontleft': 337.5,
  }

  Xtr, Xte = {}, {}
  ytr, yte = {}, {}
  ntr, nte = {}, {}

  for name, ydict in {'4x': classnums4x, '4p': classnums4p}.items():
    Xtr[name], Xte[name], ytr[name], yte[name], ntr[name], nte[name] = load(pjoin(datadir, name),
      testname='lucas', skip=['.', 'dog', 'dog2', 'doggy'], ydict=ydict, netlib=netlib
    )

  Xtr,ytr,ntr = skip_one_classers(Xtr,ytr,ntr)
  Xte,yte,nte = skip_one_classers(Xte,yte,nte)

  for name in Xtr:
    print(name)
    print("Trainset: X({}), y({})".format(Xtr[name].shape, ytr[name].shape))
    print("Testset: X({}), y({})".format(Xte[name].shape, yte[name].shape))
    print("Labels: {}".format(set(ytr[name])))

  # Merge 4x and 4p into 8
  Xtr['8'], ytr['8'], ntr['8'] = merge4to8(Xtr, ytr, ntr)
  Xte['8'], yte['8'], nte['8'] = merge4to8(Xte, yte, nte)

  #Do flip-augmentation beforehand.
  flips = [
    (classnums8['frontright'], classnums8['frontleft']),
    (classnums8['rightfront'], classnums8['leftfront']),
    (classnums8['rightback'], classnums8['leftback']),
    (classnums8['backright'], classnums8['backleft']),
    (classnums8['backleft'], classnums8['backright']),
    (classnums8['leftback'], classnums8['rightback']),
    (classnums8['leftfront'], classnums8['rightfront']),
    (classnums8['frontleft'], classnums8['frontright']),
  ]
  Xtr['8'], ytr['8'], ntr['8'] = flipall(Xtr['8'], ytr['8'], ntr['8'], flips=flips)
  Xte['8'], yte['8'], nte['8'] = flipall(Xte['8'], yte['8'], nte['8'], flips=flips)

  # Convert class-IDs into biternions.
  ytr = np.array([deg2bit(centre8[classes8[y]]) for y in ytr['8']])
  yte = np.array([deg2bit(centre8[classes8[y]]) for y in yte['8']])
  return Xtr['8'], ytr, Xte['8'], yte, nte['8']

if __name__ == '__main__':
  try:
    # Add the "models" directory to the path!
    from rospkg import RosPack
    modeldir = pjoin(RosPack().get_path('biternion'), 'models')
    sys.path.append(pjoin(RosPack().get_path('biternion'), 'scripts'))
  except ImportError:
    modeldir = os.path.dirname(os.path.abspath(os.path.join(__file__, '../models')))

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
  parser.add_argument("-m", "--modeldir",
    type=str, default=modeldir,
    help="Search-path for network description files."
  )
  parser.add_argument("-n", "--net",
    type=str, default="head_50_50",
    help="Name of the python file containing the net definition (without .py, in the `net` subfolder.)"
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

  for d in args.modeldir.split(':'):
    sys.path.append(d)
  netlib = import_module(args.net)

  printnow("Loading data from {}\n", args.datadir)
  Xtr, ytr, Xte, yte, nte = prepare_data(args.datadir, netlib)
  ytr = ytr.astype(df.floatX)
  yte = yte.astype(df.floatX)
  printnow("Got {:.2f}k training images after flipping\n", len(Xtr)/1000.0)

  aug = netlib.mkaug(Xtr, ytr)
  net = netlib.mknet()
  printnow('Network has {:.3f}M params in {} layers\n', df.utils.count_params(net)/1000.0/1000.0, len(net.modules))

  print(net[:21].forward(aug.augbatch_train(Xtr[:100])[0]).shape)

  costs = dotrain(net, crit, aug, Xtr, ytr, nepochs=args.epochs)
  print("Costs: {}".format(' ; '.join(map(str, costs))))

  dostats(net, aug, Xtr, batchsize=64)

  # Save the network.
  printnow("Saving the learned network to {}\n", args.output)
  np.save(args.output, net.__getstate__())

  # Prediction, TODO: Move to ROS node.
  s = np.argsort(nte)
  Xte,yte = Xte[s],yte[s]

  printnow("(TEMP) Doing predictions.\n", args.output)
  y_pred = dopred_bit(net, aug, Xte, batchsize=64)

  # Ensemble the flips!
  #res = maad_from_deg(bit2deg(yte), bit2deg(yte))
  res = maad_from_deg(bit2deg(y_pred), bit2deg(yte))
  printnow("MAE for test images              = {:.2f}\n", res.mean())

  #y_pred2 = ensemble_biternions([yte[::2], flipbiternions(yte[1::2])])
  y_pred2 = ensemble_biternions([y_pred[::2], flipbiternions(y_pred[1::2])])
  res = maad_from_deg(bit2deg(y_pred2), bit2deg(yte[::2]))
  printnow("MAE for flipped augmented images = {:.2f}\n", res.mean())
