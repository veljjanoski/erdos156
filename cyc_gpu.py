"""GPU exhaustive search: does Z_n (cyclic) or {0..n-1} (interval) contain a maximal Sidon set of size k?
Sidon: all differences b - b' (b != b') distinct (mod n if cyclic)  <=>  all sums b + b' (b <= b') distinct.
Maximal: every y not in B is blocked, i.e. y = b + b' - b'' (b'' distinct from b, b'; b = b' allowed) or 2y = b + b' (b != b').
Method: CPU (numba) enumerates Sidon prefixes of length `depth`; each GPU thread completes one prefix by DFS with a
difference table, and tests maximality at the leaves.  Cyclic: WLOG 0 in B.
Usage: python cyc_gpu.py cyc|int k [nmin nmax] [--depth d] [--topdown]"""
import sys, time, argparse
import numpy as np, cupy as cp
from numba import njit

KSRC = r'''
extern "C" __global__
void complete(const int* prefixes, int P, int d, int k, int n, int cyclic, int* found, int* perfect, int* witness) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= P) return;
    int B[12]; int idx[12];
    unsigned char used[1024];          // difference table: index diff (cyclic: diff mod n; interval: diff + n)
    for (int i = 0; i < 2 * n + 1 && i < 1024; i++) used[i] = 0;
    #define DI(x) (cyclic ? ((((x) % n) + n) % n) : ((x) + n))
    for (int i = 0; i < d; i++) B[i] = prefixes[(size_t)t * d + i];
    for (int i = 0; i < d; i++) for (int j = 0; j < d; j++) if (i != j) used[DI(B[i] - B[j])] = 1;
    int m = d; idx[m] = B[m-1] + 1;
    unsigned char cov[512];
    while (m >= d) {
        if (m == k) {
            for (int y = 0; y < n; y++) cov[y] = 0;
            for (int i = 0; i < k; i++) for (int j = i; j < k; j++) for (int l = 0; l < k; l++) {
                if (l == i || l == j) continue;
                int y = B[i] + B[j] - B[l];
                if (cyclic) { y %= n; if (y < 0) y += n; }
                if (y >= 0 && y < n && cov[y] < 200) cov[y] += 1;
            }
            for (int i = 0; i < k; i++) for (int j = i + 1; j < k; j++) {
                int s = B[i] + B[j];
                if (cyclic) {
                    s %= n;
                    if (n % 2 == 1) { int y = (int)(((long long)s * ((n + 1) / 2)) % n); cov[y] = 200; }
                    else if (s % 2 == 0) { cov[s / 2] = 200; cov[s / 2 + n / 2] = 200; }
                } else { if (s % 2 == 0) cov[s / 2] = 200; }
            }
            if (cyclic && n % 2 == 0) for (int i = 0; i < k; i++) cov[(B[i] + n / 2) % n] = 200;   // y + y = b + b
            for (int i = 0; i < k; i++) cov[B[i]] = 201;
            int ok = 1, perf = 1;
            for (int y = 0; y < n; y++) { if (cov[y] == 0) { ok = 0; break; } if (cov[y] != 1 && cov[y] < 200) perf = 0; }
            if (ok) {
                if (atomicExch(found, 1) == 0) for (int i = 0; i < k; i++) witness[i] = B[i];
                if (perf) { atomicExch(perfect, 1); return; }
            }
            // pop
            m--; for (int i = 0; i < m; i++) { used[DI(B[m] - B[i])] = 0; used[DI(B[i] - B[m])] = 0; } idx[m]++;
            continue;
        }
        int x = idx[m];
        if (x >= n) {
            m--; if (m >= d) { for (int i = 0; i < m; i++) { used[DI(B[m] - B[i])] = 0; used[DI(B[i] - B[m])] = 0; } idx[m]++; }
            continue;
        }
        // new differences x - B[i], B[i] - x must be unused and mutually distinct
        int good = 1;
        for (int i = 0; i < m && good; i++) {
            int a = DI(x - B[i]), b = DI(B[i] - x);
            if (used[a] || used[b] || a == b) { good = 0; break; }
            for (int j = 0; j < i; j++) { int c = DI(x - B[j]), e = DI(B[j] - x); if (a == c || a == e || b == c || b == e) { good = 0; break; } }
        }
        if (good) {
            for (int i = 0; i < m; i++) { used[DI(x - B[i])] = 1; used[DI(B[i] - x)] = 1; }
            B[m] = x; m++; if (m < k) idx[m] = x + 1;
        } else idx[m]++;
    }
}
'''
_k = cp.RawKernel(KSRC, 'complete')

@njit(cache=True)
def di(x, n, cyclic):
    return ((x % n) + n) % n if cyclic else x + n

@njit(cache=True)
def prefixes(k, n, cyclic, depth, out):
    B = np.zeros(12, dtype=np.int64); idx = np.zeros(12, dtype=np.int64)
    used = np.zeros(2 * n + 1, dtype=np.uint8)
    cnt = 0
    if cyclic: B[0] = 0; m = 1; idx[1] = 1; mlow = 1
    else: m = 0; idx[0] = 0; mlow = 0
    while m >= mlow:
        if m == depth:
            if cnt >= out.shape[0] - 1: return cnt   # buffer full: caller asserts
            for i in range(depth): out[cnt, i] = B[i]
            cnt += 1
            m -= 1
            for i in range(m): used[di(B[m] - B[i], n, cyclic)] = 0; used[di(B[i] - B[m], n, cyclic)] = 0
            idx[m] += 1
            continue
        x = idx[m]
        if x >= n - (k - 1 - m):
            m -= 1
            if m >= mlow:
                for i in range(m): used[di(B[m] - B[i], n, cyclic)] = 0; used[di(B[i] - B[m], n, cyclic)] = 0
                idx[m] += 1
            continue
        good = True
        for i in range(m):
            a = di(x - B[i], n, cyclic); b = di(B[i] - x, n, cyclic)
            if used[a] or used[b] or a == b: good = False; break
            for j in range(i):
                c = di(x - B[j], n, cyclic); e = di(B[j] - x, n, cyclic)
                if a == c or a == e or b == c or b == e: good = False; break
            if not good: break
        if good:
            for i in range(m): used[di(x - B[i], n, cyclic)] = 1; used[di(B[i] - x, n, cyclic)] = 1
            B[m] = x; m += 1
            if m < depth: idx[m] = x + 1
        else: idx[m] += 1
    return cnt

def exists(k, n, cyclic, depth):
    assert n <= 511
    buf = np.zeros((4_000_000, depth), dtype=np.int32)
    P = prefixes(k, n, cyclic, depth, buf)
    assert P < buf.shape[0] - 1, 'prefix buffer too small'
    if P == 0: return False, False, None
    pre = cp.asarray(buf[:P].ravel())
    found = cp.zeros(1, dtype=cp.int32); perfect = cp.zeros(1, dtype=cp.int32); wit = cp.zeros(k, dtype=cp.int32)
    _k(((P + 127) // 128,), (128,), (pre, np.int32(P), np.int32(depth), np.int32(k), np.int32(n), np.int32(cyclic), found, perfect, wit))
    cp.cuda.Device().synchronize()
    f = bool(found.get()[0])
    return f, bool(perfect.get()[0]), (wit.get().tolist() if f else None)

if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('mode'); ap.add_argument('k', type=int)
    ap.add_argument('nmin', type=int, nargs='?', default=None); ap.add_argument('nmax', type=int, nargs='?', default=None)
    ap.add_argument('--depth', type=int, default=3); ap.add_argument('--topdown', action='store_true')
    a = ap.parse_args(); cyclic = a.mode == 'cyc'; k = a.k
    nmax = a.nmax or (k**3 - k*k + 2*k) // 2 + 1; nmin = a.nmin or k
    best = 0; bestp = 0; t0 = time.time()
    rng = range(nmax, nmin - 1, -1) if a.topdown else range(nmin, nmax + 1)
    for n in rng:
        t = time.time(); f, p, w = exists(k, n, cyclic, a.depth)
        if f: best = max(best, n)
        if p: bestp = max(bestp, n)
        if f or n % 10 == 0: print(f"  n={n}: maximal={f} perfect={p} witness={w} [{time.time()-t:.1f}s]", flush=True)
        if a.topdown and f: break
    print(f"{a.mode} k={k}: max n with maximal size-k Sidon set = {best}; largest n with a perfect one = {bestp}  [{time.time()-t0:.0f}s]")
