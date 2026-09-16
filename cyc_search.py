"""For each k, find the largest n such that Z_n (cyclic) / [0,n) (interval) contains a maximal Sidon set of size k.
Also record whether a 'perfect' set exists (every x not in B has exactly one representation b+b'-b'', b'!=b'', as unordered pair).
DFS over Sidon sets containing 0 (cyclic: also fix translation), numba-compiled."""
import sys, time
import numpy as np
from numba import njit

@njit(cache=True)
def is_sidon_add(B, m, x, n, cyclic):
    # can x be added to B[:m] keeping Sidon (differences distinct, mod n if cyclic)?
    # check new differences x-B[i] (and B[i]-x) against old differences and each other
    for i in range(m):
        d1 = x - B[i]
        if cyclic: d1 %= n
        for j in range(m):
            if j == i: continue
            d2 = B[j] - B[i]
            if cyclic: d2 %= n
            if d1 == d2: return False
            d3 = B[i] - x
            if cyclic: d3 %= n
            if d3 == d2: return False
        for j in range(i + 1, m):
            d2 = x - B[j]
            if cyclic: d2 %= n
            if d1 == d2: return False
            d3 = B[j] - x
            if cyclic: d3 %= n
            if d1 == d3: return False
    return True

@njit(cache=True)
def coverage(B, k, n, cyclic, cov):
    for x in range(n): cov[x] = 0
    for i in range(k):
        for j in range(i, k):          # unordered pair {i,j} (i==j allowed: midpoint-type x with 2x = b+b' handled below)
            for l in range(k):
                if l == i or l == j: continue
                x = B[i] + B[j] - B[l]
                if cyclic: x %= n
                if 0 <= x < n: cov[x] += 1
    # midpoint condition: x with 2x = b_i + b_j, i != j  (only for x not in B)
    for i in range(k):
        for j in range(i + 1, k):
            s = B[i] + B[j]
            if cyclic:
                # solve 2x = s mod n
                for x in range(n):
                    if (2 * x - s) % n == 0: cov[x] += 100
            else:
                if s % 2 == 0: cov[s // 2] += 100
    for i in range(k): cov[B[i]] = 1000

@njit(cache=True)
def search(k, n, cyclic):
    """returns (exists_maximal, exists_perfect)"""
    B = np.zeros(k, dtype=np.int64); cov = np.zeros(n, dtype=np.int64)
    idx = np.zeros(k, dtype=np.int64)
    if cyclic:
        B[0] = 0; m = 1; idx[1] = 1; mlow = 1
    else:
        m = 0; idx[0] = 0; mlow = 0
    found = False; perfect = False
    while m >= mlow:
        if m == k:
            coverage(B, k, n, cyclic, cov)
            ok = True; perf = True
            for x in range(n):
                if cov[x] == 0: ok = False; break
                if cov[x] != 1 and cov[x] < 100: perf = False
            if ok:
                found = True
                if perf: perfect = True; return found, perfect
            m -= 1; idx[m] += 1
            continue
        x = idx[m]
        if x >= n:
            m -= 1
            if m >= mlow: idx[m] += 1
            continue
        if is_sidon_add(B, m, x, n, cyclic):
            B[m] = x; m += 1
            if m < k: idx[m] = x + 1
        else:
            idx[m] += 1
    return found, perfect

if __name__ == '__main__':
    cyclic = sys.argv[1] == 'cyc'; kmax = int(sys.argv[2])
    for k in range(2, kmax + 1):
        t0 = time.time(); best = 0; bestp = 0
        nmax = (k**3 - k*k + 2*k) // 2 + 1
        for n in range(k, nmax + 1):
            f, p = search(k, n, cyclic)
            if f: best = n
            if p: bestp = n
        print(f"{'cyclic' if cyclic else 'interval'} k={k}: max n with maximal size-k Sidon set = {best}; largest n with a perfect one = {bestp}  [{time.time()-t0:.0f}s]", flush=True)
