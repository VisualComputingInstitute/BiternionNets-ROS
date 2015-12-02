import DeepFried2 as df

class Flatten(df.Module):
  def symb_forward(self, symb_in):
    return symb_in.flatten(2)


class Biternion(df.Module):
  def symb_forward(self, symb_in):
    return symb_in / df.T.sqrt((symb_in**2).sum(axis=1, keepdims=True))


class BiternionCriterion(df.Criterion):
  def __init__(self, kappa=None):
    """
    Setting `kappa` to `None` uses the cosine criterion,
    setting it to any other numeric value uses von-Mises.
    """
    df.Criterion.__init__(self)
    self.kappa = kappa

  def symb_forward(self, symb_in, symb_tgt):
    # For normalized `symb_in` and `symb_tgt`, dot-product (batched)
    # outputs a cosine value, i.e. between -1 (worst) and 1 (best)
    cos_angles = df.T.batched_dot(symb_in, symb_tgt)

    # This is the only difference to the pure `CosineCriterion`.
    if self.kappa is not None:
        cos_angles = df.T.exp(self.kappa * (cos_angles - 1))

    return df.T.mean(1 - cos_angles)
