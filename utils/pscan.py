import math
import torch
import torch.nn.functional as F

"""
Parallel Scan (PScan) Blelloch algorithm in pure PyTorch.
From alxndrTL/mamba.py (MIT License).
Computes prefix scan H[t] = A[t] * H[t-1] + X[t] in O(log L) parallel steps on GPU/CPU
with custom autograd backward pass, completely eliminating Python sequential for-loops.
"""

def npo2(len):
    return 2 ** math.ceil(math.log2(max(1, len)))

def pad_npo2(X):
    len_npo2 = npo2(X.size(2))
    pad_tuple = (0, 0, 0, len_npo2 - X.size(2))
    return F.pad(X, pad_tuple, "constant", 0)

class PScan(torch.autograd.Function):
    @staticmethod
    def pscan(A, X):
        B, D, L, _ = A.size()
        num_steps = int(math.log2(L))

        Aa = A
        Xa = X
        for _ in range(num_steps - 2):
            T = Xa.size(2)
            Aa = Aa.view(B, D, T // 2, 2, -1)
            Xa = Xa.view(B, D, T // 2, 2, -1)

            Xa[:, :, :, 1].add_(Aa[:, :, :, 1].mul(Xa[:, :, :, 0]))
            Aa[:, :, :, 1].mul_(Aa[:, :, :, 0])

            Aa = Aa[:, :, :, 1]
            Xa = Xa[:, :, :, 1]

        if Xa.size(2) == 4:
            Xa[:, :, 1].add_(Aa[:, :, 1].mul(Xa[:, :, 0]))
            Aa[:, :, 1].mul_(Aa[:, :, 0])

            Xa[:, :, 3].add_(Aa[:, :, 3].mul(Xa[:, :, 2] + Aa[:, :, 2].mul(Xa[:, :, 1])))
        elif Xa.size(2) == 2:
            Xa[:, :, 1].add_(Aa[:, :, 1].mul(Xa[:, :, 0]))
            return
        else:
            return

        Aa = A[:, :, 2**(num_steps - 2) - 1:L:2**(num_steps - 2)]
        Xa = X[:, :, 2**(num_steps - 2) - 1:L:2**(num_steps - 2)]
        Xa[:, :, 2].add_(Aa[:, :, 2].mul(Xa[:, :, 1]))
        Aa[:, :, 2].mul_(Aa[:, :, 1])

        for k in range(num_steps - 3, -1, -1):
            Aa = A[:, :, 2**k - 1:L:2**k]
            Xa = X[:, :, 2**k - 1:L:2**k]

            T = Xa.size(2)
            Aa = Aa.view(B, D, T // 2, 2, -1)
            Xa = Xa.view(B, D, T // 2, 2, -1)

            Xa[:, :, 1:, 0].add_(Aa[:, :, 1:, 0].mul(Xa[:, :, :-1, 1]))
            Aa[:, :, 1:, 0].mul_(Aa[:, :, :-1, 1])

    @staticmethod
    def pscan_rev(A, X):
        B, D, L, _ = A.size()
        num_steps = int(math.log2(L))

        Aa = A
        Xa = X
        for _ in range(num_steps - 2):
            T = Xa.size(2)
            Aa = Aa.view(B, D, T // 2, 2, -1)
            Xa = Xa.view(B, D, T // 2, 2, -1)

            Xa[:, :, :, 0].add_(Aa[:, :, :, 0].mul(Xa[:, :, :, 1]))
            Aa[:, :, :, 0].mul_(Aa[:, :, :, 1])

            Aa = Aa[:, :, :, 0]
            Xa = Xa[:, :, :, 0]

        if Xa.size(2) == 4:
            Xa[:, :, 2].add_(Aa[:, :, 2].mul(Xa[:, :, 3]))
            Aa[:, :, 2].mul_(Aa[:, :, 3])

            Xa[:, :, 0].add_(Aa[:, :, 0].mul(Xa[:, :, 1] + Aa[:, :, 1].mul(Xa[:, :, 2])))
        elif Xa.size(2) == 2:
            Xa[:, :, 0].add_(Aa[:, :, 0].mul(Xa[:, :, 1]))
            return
        else:
            return

        Aa = A[:, :, 2**(num_steps - 2) - 1:L:2**(num_steps - 2)]
        Xa = X[:, :, 2**(num_steps - 2) - 1:L:2**(num_steps - 2)]
        Xa[:, :, 1].add_(Aa[:, :, 1].mul(Xa[:, :, 2]))
        Aa[:, :, 1].mul_(Aa[:, :, 2])

        for k in range(num_steps - 3, -1, -1):
            Aa = A[:, :, 2**k - 1:L:2**k]
            Xa = X[:, :, 2**k - 1:L:2**k]

            T = Xa.size(2)
            Aa = Aa.view(B, D, T // 2, 2, -1)
            Xa = Xa.view(B, D, T // 2, 2, -1)

            Xa[:, :, :-1, 1].add_(Aa[:, :, :-1, 1].mul(Xa[:, :, 1:, 0]))
            Aa[:, :, :-1, 1].mul_(Aa[:, :, 1:, 0])

    @staticmethod
    def forward(ctx, A, X):
        ctx.save_for_backward(A, X)
        A_ = A.clone()
        X_ = X.clone()
        PScan.pscan(A_, X_)
        return X_

    @staticmethod
    def backward(ctx, grad_output):
        A, X = ctx.saved_tensors
        B, D, L, _ = A.size()

        Q = torch.zeros_like(X)
        Q[:, :, :-1] = grad_output[:, :, 1:] * A[:, :, 1:]
        PScan.pscan_rev(A, Q)
        Q[:, :, 1:] += grad_output[:, :, 1:]

        grad_A = torch.zeros_like(A)
        grad_A[:, :, 1:] = Q[:, :, 1:] * X[:, :, :-1]

        grad_X = Q
        return grad_A, grad_X

def pscan(A, X):
    """
    A: (B, D, L, N)
    X: (B, D, L, N)
    Returns: (B, D, L, N)
    """
    L = A.size(2)
    L_pad = npo2(L)
    if L_pad != L:
        A = pad_npo2(A)
        X = pad_npo2(X)
    out = PScan.apply(A, X)
    if L_pad != L:
        out = out[:, :, :L]
    return out
