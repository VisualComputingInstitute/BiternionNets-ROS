import cv2
import numpy as np

import torch
import torch.nn as nn
from torch.autograd import Variable

import common as C


# Monkey-patch because I trained with a newer version.
# This can be removed once PyTorch 0.4.x is out.
import torch._utils
try:
    torch._utils._rebuild_tensor_v2
    OLD_TORCH = False
except AttributeError:
    def _rebuild_tensor_v2(storage, storage_offset, size, stride, requires_grad, backward_hooks):
        tensor = torch._utils._rebuild_tensor(storage, storage_offset, size, stride)
        tensor.requires_grad = requires_grad
        tensor._backward_hooks = backward_hooks
        return tensor
    torch._utils._rebuild_tensor_v2 = _rebuild_tensor_v2
    OLD_TORCH = True


def conv3(cin, cout, **kw):
    kw.setdefault('bias', False)
    kw.setdefault('padding', 1)
    return nn.Conv2d(cin, cout, 3, **kw)


class Model:
    def __init__(self, weightsname, GPU=False, hfactor=0.6, wfactor=1.0):
        self._GPU = GPU
        self._hfactor = hfactor
        self._wfactor = wfactor

        torch.backends.cudnn.benchmark = True  # Run benchmark to select fastest implementation of ops.

        # TODO: Load this from saved model settings.
        pool = 'max'
        dropout = (0.2, 0.2)

        if pool == 'avg':
            head = nn.AvgPool2d((1,5))
        elif pool == 'max':
            head = nn.MaxPool2d((1,5))
        elif pool == 'fc':
            head = nn.Sequential(nn.Conv2d(128, 128, (1,5), bias=False), nn.BatchNorm2d(128), nn.ReLU())

        self._net = nn.Sequential(  # 50x80 -> 40x72
            conv3( 3, 16), nn.BatchNorm2d(16), nn.ReLU(),
            conv3(16, 32), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2),  # 20x36
            (nn.Dropout2d(dropout[0]) if dropout is not None else nn.Sequential()),
            conv3(32, 32), nn.BatchNorm2d(32), nn.ReLU(),
            conv3(32, 64), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2),  # 10x18
            (nn.Dropout2d(dropout[0]) if dropout is not None else nn.Sequential()),
            conv3(64, 64), nn.BatchNorm2d(64), nn.ReLU(),
            conv3(64, 96), nn.BatchNorm2d(96), nn.ReLU(),
            nn.MaxPool2d(2),  # 5x9
            (nn.Dropout2d(dropout[0]) if dropout is not None else nn.Sequential()),
            nn.Conv2d(96, 96, 3), nn.BatchNorm2d(96), nn.ReLU(),
            nn.Conv2d(96,128, 3), nn.BatchNorm2d(128), nn.ReLU(),
            head,
            (nn.Dropout(dropout[1]) if dropout is not None else nn.Sequential()),
            nn.Conv2d(128, 3, 1)
        )

        # Load the network weights onto the CPU first, no matter where they trained.
        checkpoint = torch.load(weightsname, map_location='cpu')
        self._net.load_state_dict(checkpoint)

        # And finally move the thing to the GPU, if needed.
        self._net = self._maybe_gpu(self._net)

        self._net.eval()

    def __call__(self, rgb, d, detrects):
        if len(detrects) == 0:
            return np.array([]), np.array([])

        images = np.array([self._preproc(*self._cutout(rgb, d, *rect)) for rect in detrects])

        preds = self._forward(images)

        # Need to split out the raw predictions into what they really are,
        # and apply their respecitve non-linearities.
        biternions, confidences = preds[:,:2], preds[:,2]
        biternions = C.normalized(biternions, axis=1)
        print(biternions[0,:,0,0])
        confidences = self._conf_nonlin(confidences)

        # The predictions still contain the spatial dimension: BCHW.
        # We just average these out, though more advanced things could be done,
        # such as weighted average weighting by certainty, etc.
        biternions = np.mean(biternions, axis=(-2,-1))
        confidences = np.mean(confidences, axis=(-2,-1))

        # Need to normalize again in order to have real biternions
        biternions = C.normalized(biternions)

        # Convert the biternions into an angle as a common API.
        angles = C.bit2deg(biternions)

        return angles, confidences

    def getrect(self, x, y, w, h):
        """
        Transform an original detection/tracking box into what we need for
        making predictions, i.e. should be the same as during dumping.
        """
        x, y, w, h = C.cutout_hwfact(x, y, w, h, self._hfactor, self._wfactor)
        return x, y, w, h

    def _cutout(self, rgb, d, x, y, w, h):
        rect = self.getrect(x, y, w, h)
        return C.cutout(rgb, *rect), C.cutout(d, *rect)

    def _preproc(self, rgb, d):
        """ Transforms an rgb+d cut-out into a network-input.

        We could also do background-subtraction here, but I trained the model without.
        """
        # First, resize the image to a fixed size.
        # Note the argument is (w,h) but the shape is HWC
        rgb = cv2.resize(rgb, (80, 50))
        # From HWC to CHW, and then [0,255] to [0,1]
        return rgb.transpose(2,0,1).astype(np.float32)/255

    def _conf_nonlin(self, k):
        MAXK = 8
        return MAXK*C.sigmoid(float(np.log(MAXK-1))/(MAXK/2)*(k-(MAXK/2)))

    def _forward(self, x):
        """ Send numpy-array x through the network and return numpy-array of results. """
        if OLD_TORCH:
            x_var = Variable(self._maybe_gpu(torch.from_numpy(x)), volatile=True)
            return self._net(x_var).data.cpu().numpy()
        else:
            with torch.no_grad():
                x_var = Variable(self._maybe_gpu(torch.from_numpy(x)))
                return self._net(x_var).data.cpu().numpy()

    def _maybe_gpu(self, whatever):
        """ Moves `what` to CUDA and returns it, if `use_cuda` and it's available.

        Actually, `use_cuda` is the GPU-index to be used, which means `0` uses the
        first GPU. To not use GPUs, set `use_cuda` to `False` instead.
        """
        if self._GPU is not False and torch.cuda.is_available():
            whatever = whatever.cuda(device=self._GPU, **kw)
        return whatever
