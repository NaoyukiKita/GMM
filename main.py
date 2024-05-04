from models import EMGMM, VBGMM
from util.data import make_data

sizes = [800, 1000, 1200]
means = [
    [-2, 0],
    [10, -6],
    [8, 9]
]
vars = [
    [[1, 2],
     [2, 9]],
    [[2, 1],
     [1, 3]],
    [[2, 1],
     [1, 2]]
]
sample_data = make_data(sizes, means, vars)

K = len(sizes)
N = sum(sizes)

em = EMGMM(n_components=K)
vb = VBGMM(n_components=K)

em.fit(sample_data)
vb.fit(sample_data)

print("=====EM-GMM=====")
for k in range(K):
    print(f"size: {em.pi[k] * N}")
    print(f"mean:\n{em.mu[k]}")
    print(f"variance-covariance matrix:\n{em.Sigma[k]}\n")

print("\n\n=====VB-GMM=====")
for k in range(K):
    print(f"size: {vb.pi[k] * N}")
    print(f"mean:\n{vb.mu[k]}")
    print(f"variance-covariance matrix:\n{vb.Sigma[k]}\n")