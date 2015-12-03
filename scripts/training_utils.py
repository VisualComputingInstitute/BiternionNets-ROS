import numpy as np
import DeepFried2 as df
from lbtoolbox.util import batched

def dotrain(model, crit, aug, Xtr, ytr, nepochs=3, batchsize=100, title=None):
    opt = df.AdaDelta(rho=.95, eps=1e-7, lr=1)
    model.training()
    costs = []
    print("Training in progress...")
    for e in range(nepochs):
        print("Current epoch: {0} out of {1}".format(e+1,nepochs))
        batchcosts = []
        for Xb, yb in batched(batchsize, Xtr, ytr, shuf=True):
            if aug is not None:
                Xb, yb = aug.augbatch_train(Xb, yb)
            model.zero_grad_parameters()
            cost = model.accumulate_gradients(Xb, yb, crit)
            opt.update_parameters(model)
            batchcosts.append(cost)

        costs.append(np.mean(batchcosts))
    return costs


def dostats(model, aug, Xtr, batchsize=100):
    model.training()

    for Xb in batched(batchsize, Xtr):
        if aug is None:
            model.accumulate_statistics(Xb)
        else:
            for Xb_aug in aug.augbatch_pred(Xb):
                model.accumulate_statistics(Xb_aug)

def dopred(model, aug, X, ensembling, output2preds, batchsize=100):
    model.evaluate()
    y_preds = []
    for Xb in batched(batchsize, X):
        if aug is None:
            p_y = model.forward(X)
        else:
            p_y = ensembling([model.forward(X) for X in aug.augbatch_pred(Xb)])
        y_preds += list(output2preds(p_y))
    return np.array(y_preds)
