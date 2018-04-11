import numpy as np
import common as C

class ModelV1:
    def __call__(self, rgb, d, detrects):
        if len(detrects) == 0:
            return np.array([]), np.array([])

        images = []
        for detrect in detrects:
            detrect = self.getrect(*detrect)
            images.append(self._preproc(C.cutout(rgb, *detrect), C.cutout(d, *detrect)))
        images = np.array(images)

        bits = [self._net.forward(batch) for batch in self._aug.augbatch_pred(images, fast=True)]
        preds = C.bit2deg(C.ensemble_biternions(bits)) - 90  # Subtract 90 to correct for "my weird" origin.

        return preds, np.full(len(detrects), 0.83)
