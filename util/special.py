import numpy as np
from numpy import log

def dig(x: float):
    if x < 0:
        print(f"UnexpectedArgumentWarning: digamma expects non-negative variable, but gets {x}.")
    r = 0
    while (x<=5):
        r -= 1/x
        x += 1
    f = 1/(x*x)
    t = f*(-1/12.0 + f*(1/120.0 + f*(-1/252.0 + f*(1/240.0 + f*(-1/132.0
        + f*(691/32760.0 + f*(-1/12.0 + f*3617/8160.0)))))))
    return r + log(x) - 0.5/x + t

def digamma(x: np.array):
    return np.vectorize(dig)(x)