from numpy import abs, exp, power, sqrt
from numpy import pi as PI
from numpy.linalg import det, inv

def norm_pdf(D, x, m, sigma):
    diff = x - m
    kernel = - diff.T @ inv(sigma) @ diff / 2
    q = power(2*PI, D/2) * sqrt(abs(det(sigma)))

    return exp(kernel) / q
