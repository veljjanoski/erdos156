"""Independent check, written from the definitions (sums, not differences):
 - Sidon in Z_n: all sums a+b (a<=b, from B) distinct mod n.
 - maximal: for every y not in B, B+{y} is not Sidon.
(1) verify every witness printed by cyc_gpu.py;  (2) brute force k<=5 all n (every k-subset containing 0);
(3) brute force nonexistence for k=6 at n=81..97 and k=7 at n=121..126 via numba combination enumeration."""
import re, itertools, sys, time
import numpy as np
from numba import njit

def sidon(B, n):
    s = set()
    for i in range(len(B)):
        for j in range(i, len(B)):
            v = (B[i] + B[j]) % n
            if v in s: return False
            s.add(v)
    return True
def maximal(B, n):
    return sidon(B, n) and all(not sidon(B + [y], n) for y in range(n) if y not in B)

# (1) witnesses
for f in ('cyc7.log',):
    for line in open(f):
        m = re.search(r"n=(\d+): maximal=True.*witness=\[([\d, ]+)\]", line)
        if m:
            n = int(m.group(1)); B = [int(x) for x in m.group(2).split(',')]
            assert maximal(B, n), (n, B)
print("all k=7 witnesses verified by definition")
for n, B in [(80, [0, 43, 47, 48, 50, 68]), (120, [0, 4, 28, 45, 46, 48, 57])]:
    print(n, B, 'maximal Sidon in Z_n:', maximal(B, n))

# (2) brute force k<=5
for k in range(2, 6):
    best = 0
    for n in range(k, (k**3 - k*k + 2*k)//2 + 2):
        if any(maximal([0] + list(c), n) for c in itertools.combinations(range(1, n), k - 1)): best = n
    print(f"brute force cyclic k={k}: max n = {best}")

@njit(cache=True)
def exists_bf(k, n):
    # enumerate combinations of k-1 elements from 1..n-1 (plus 0), test by sums
    c = np.arange(1, k, dtype=np.int64)     # c[0..k-2]
    B = np.zeros(k, dtype=np.int64); sums = np.zeros(n, dtype=np.int64)
    while True:
        B[0] = 0
        for i in range(k - 1): B[i + 1] = c[i]
        # sidon by sums
        for y in range(n): sums[y] = 0
        ok = True
        for i in range(k):
            for j in range(i, k):
                v = (B[i] + B[j]) % n
                if sums[v]: ok = False
                sums[v] = 1
        if ok:
            # maximal: every y not in B creates a repeated sum: y+b in sums or 2y in sums or y+b == y+b' (impossible) ...
            # B+{y} not Sidon iff exists b in B with (y+b) mod n in sums, or (2y) mod n in sums, or y+b == y+b' (no)
            allblocked = True
            for y in range(n):
                inB = False
                for i in range(k):
                    if B[i] == y: inB = True
                if inB: continue
                blocked = sums[(2 * y) % n] == 1
                if not blocked:
                    for i in range(k):
                        if sums[(y + B[i]) % n] == 1: blocked = True; break
                if not blocked: allblocked = False; break
            if allblocked: return True
        # next combination
        i = k - 2
        while i >= 0 and c[i] == n - (k - 1) + i: i -= 1
        if i < 0: return False
        c[i] += 1
        for j in range(i + 1, k - 1): c[j] = c[j - 1] + 1

if __name__ == '__main__':
    t = time.time()
    print("brute force k=6, n=80:", exists_bf(6, 80), "n=81..97:", [n for n in range(81, 98) if exists_bf(6, n)], f"[{time.time()-t:.0f}s]", flush=True)
    t = time.time()
    print("brute force k=7, n=120:", exists_bf(7, 120), "n=121..123:", [n for n in range(121, 124) if exists_bf(7, n)], f"[{time.time()-t:.0f}s]", flush=True)
    