import numpy as np
import cv2
import sys,os
import argparse

import DeepFried2 as df
from training_utils import dotrain, dostats, dopred
import lbtoolbox.util as lbu
from lbtoolbox.util import batched
from lbtoolbox.thutil import count_params
from lbtoolbox.augmentation import AugmentationPipeline, Cropper

import theano as th
import theano.tensor as T

floatX = th.config.floatX
pjoin = os.path.join
reportsdir = '/work/kurin/reports/vision-heads2'
datadir = '/work/kurin/vision-heads2'

class Flatten(df.Module):
  def symb_forward(self, symb_in):
    return symb_in.flatten(2)

def myimread(fname, zeroone=True, resize=None):
  im = cv2.imread(fname, flags=cv2.IMREAD_COLOR)
  if im is None:
    raise ValueError("Couldn't load image " + fname)

  if resize is not None:
    im = cv2.resize(im, resize, interpolation=cv2.INTER_LANCZOS4)

  # In OpenCV, color dimension is last, but theano likes it to be first.
  # (That's map of triplets vs three maps philosophy.)
  im = np.rollaxis(im, 2, 0) #TODO add comment or vars instead of bare numbers
  return im.astype(np.float32)/256 if zeroone else im #TODO remove hardcoded values

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
  
  # Sorted testing stuff is better to look at.
  s = np.argsort(nte)

  return np.array(Xtr), np.array(Xte)[s], np.array(ytr), np.array(yte)[s], ntr, [nte[i] for i in s]

def merge(X, y, n):
  y8 = np.full_like(y['4p'], np.nan)
  for idx4p, n4p in enumerate(n['4p']):
    idx4x = n['4x'].index(n4p)
    y4x, y4p = y['4x'][idx4x], y['4p'][idx4p]
    y8[idx4p] = y4x + y4p if not (y4x == 0 and y4p == 3) else 7
  return X['4p'].copy(), y8, n['4p'].copy()

def flipped(X, y, n, old, new):
    indices = np.where(y == old)[0]
    return lbu.flipany(X[indices], dim=3), np.full(len(indices), new, dtype=y.dtype), [n[i] for i in indices]

def flipall(X, y, n, flips):
  fx, fy, fn = [], [], []
  for old, new in flips:
    a, b, c = flipped(X, y, n, old, new)
    fx.append(a) ; fy.append(b) ; fn.append(c)
  return np.concatenate([X] + fx), np.concatenate([y] + fy), n + sum(fn, list())

class VonMisesBiternionCriterion(df.Criterion):
  def __init__(self, kappa):
    df.Criterion.__init__(self)
    self.kappa = kappa

  def symb_forward(self, symb_in, symb_tgt):
    cos_angles = df.T.batched_dot(symb_in, symb_tgt)

    # This is the only difference to the pure `CosineCriterion`.
    # Obviously, they could be in the same class, but I separate them here for narration.
    cos_angles = df.T.exp(self.kappa * (cos_angles - 1))

    return df.T.mean(1 - cos_angles)

class CosineCriterion(df.Criterion):
  def symb_forward(self, symb_in, symb_tgt):
    # For normalized `symb_in` and `symb_tgt`, dot-product (batched)
    # outputs a cosine value, i.e. between -1 (worst) and 1 (best)
    cos_angles = df.T.batched_dot(symb_in, symb_tgt)

    # Rescale to a cost going from 2 (worst) to 0 (best) each, then take mean.
    return df.T.mean(1 - cos_angles)

class Biternion(df.Module):
  def symb_forward(self, symb_in):
    return symb_in / df.T.sqrt((symb_in**2).sum(axis=1, keepdims=True))

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

def save_angle(img, pred, name, zf=5):
  viz = alexviz.visualize_head_pose_regression(img[::-1,:,:], pred, circular=True, zoom_factor=zf, val_bg=0.45)
  bgra01c = np.rollaxis(viz, 0, 3)[:,:,[2,1,0,3]]
  path = pjoin(reportsdir, name + '.png')
  os.makedirs(os.path.dirname(path), exist_ok=True)
  cv2.imwrite(path, (255*bgra01c).astype(np.uint8))

def ensemble_degrees(angles):
  return np.arctan2(np.mean(np.sin(np.deg2rad(angles)), axis=0), np.mean(np.cos(np.deg2rad(angles)), axis=0))

def dopred_deg(model, aug, X, batchsize=100):
  return np.rad2deg(dopred(model, aug, X, ensembling=ensemble_degrees, output2preds=lambda x: x, batchsize=batchsize))

def maad_from_deg(preds, reals):
  return np.rad2deg(np.abs(np.arctan2(np.sin(np.deg2rad(reals-preds)), np.cos(np.deg2rad(reals-preds)))))

def show_errs_deg(preds, reals, epoch=-1):
    errs = maad_from_deg(preds, reals)
    mean_errs = np.mean(errs, axis=1)
    std_errs = np.std(errs, axis=1)
    #add degree sign here
    print("Error: {:5.2f}+-{:5.2f}".format(np.mean(mean_errs), np.mean(std_errs)))
    print("Stdev: {:5.2f}+-{:5.2f}".format(np.std(mean_errs), np.std(std_errs)))

def prepare_data():
  classes4x = ['front','right','back','left']
  classnums4x = {c: i for i, c in enumerate(classes4x)}
  classes4p = ['frontright','backright','backleft','frontleft']
  classnums4p = {c: i for i, c in enumerate(classes4p)}
  classes8 = ['frontright','rightfront','rightback','backright','backleft','leftback','leftfront','frontleft']
  classnums8 = {c: i for i, c in enumerate(classes8)}

  centre4_deg = {
      'front': 0,
      'frontright': 45,
      'right': 90,
      'backright': 135,
      'back': 180,
      'backleft': 225,
      'left': 270,
      'frontleft': 315,
  }
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

  centre4_rad = {k: np.deg2rad(v) for k, v in centre4_deg.items()}
  centre8_rad = {k: np.deg2rad(v) for k, v in centre8_deg.items()}
  centre4_vec = {k:    deg2bit(v) for k, v in centre4_deg.items()}
  centre8_vec = {k:    deg2bit(v) for k, v in centre8_deg.items()}

  Xtr, Xte = {}, {}
  ytr, yte = {}, {}
  ntr, nte = {}, {}
  
  for name, ydict in {'4x': classnums4x, '4p': classnums4p}.items():
    Xtr[name], Xte[name], ytr[name], yte[name], ntr[name], nte[name] = load(pjoin(datadir, name), 
      testname='lucas', skip=['dog', 'dog2', 'doggy'], ydict=ydict
    )
 
  
  for name in Xtr:
    print(name)
    print("Trainset: X({}), y({})".format(Xtr[name].shape, ytr[name].shape))
    print("Testset: X({}), y({})".format(Xte[name].shape, yte[name].shape))
    print("Random label: {}".format(set(ytr[name])))

  Xtr['8'], ytr['8'], ntr['8'] = merge(Xtr, ytr, ntr)
  
  
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

  Xtr['8'], ytr['8'], ntr['8'] = flipall(Xtr['8'], ytr['8'], ntr['8'], flips=[
      (classnums8['frontleft'], classnums8['frontright']),
      (classnums8['frontright'], classnums8['frontleft']),
      (classnums8['leftfront'], classnums8['rightfront']),
      (classnums8['rightfront'], classnums8['leftfront']),
      (classnums8['backleft'], classnums8['backright']),
      (classnums8['backright'], classnums8['backleft']),
      (classnums8['leftback'], classnums8['rightback']),
      (classnums8['rightback'], classnums8['leftback']),
  ])
  
  Xte['8'], yte['8'], nte['8'] = merge(Xte, yte, nte)
  ytr = np.array([centre8_vec[classes8[y]]for y in ytr['8']])
  yte = np.array([centre8_vec[classes8[y]]for y in yte['8']])
  
  return Xtr['8'], ytr, ntr['8'], Xte['8'], yte

if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Head angle biternion net training')
  parser.add_argument(
    "-t",
    "--type", 
    type=str, 
    default='cosine', 
    help='net type: "cosine" or "biternion" ',
  )

  types = ['cosine','von-mises']
  type = vars(parser.parse_args())['type']
  print(type+" net will be used")
  if type not in types:
    print("You specified wrong net type. Sorry =(")
    sys.exit()  

  Xtr,ytr,ntr,Xte,yte = prepare_data()
  Xtr, ytr = Xtr, ytr.astype(df.floatX)
  Xte, yte = Xte, yte.astype(df.floatX)
  aug = AugmentationPipeline(Xtr, ytr, Cropper((46,46)))
 
  if type == 'cosine':  
    net = mknet(df.Linear(512, 2, initW=df.init.normal(0.01)), Biternion())
    print('{:.3f}M params'.format(count_params(net)/1000000))
    trains_linreg_bt_cos = dotrain(net, CosineCriterion(), aug, Xtr, ytr)
    print("Costs: {}".format(' ; '.join(map(str, trains_linreg_bt_cos))))
    dostats(net, aug, Xtr, batchsize=1000)
    y_pred = bit2deg(dopred_deg(net, aug, Xte))
    res = maad_from_deg(y_pred, bit2deg(yte))
    print(res.mean())
  elif type == 'von-mises':
    net_vm = mknet(df.Linear(512, 2, initW=df.init.normal(0.01)), Biternion())
    print('{:.3f}M params'.format(count_params(net_vm)/1000000))
    trains__bt_vm = dotrain(net_vm, VonMisesBiternionCriterion(1), aug, Xtr, ytr)
    dostats(net_vm, aug, Xtr, batchsize=1000)
    y_pred_bt_vm = bit2deg(dopred_deg(net_vm, aug, Xte))
    res = maad_from_deg(y_pred_bt_vm, bit2deg(yte))
    print(res.mean())
  
#TODO we can estimate correspondance between orig and flipped img pred
#if they agree, we are bit-scalable, if not -> =(
