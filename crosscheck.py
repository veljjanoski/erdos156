"""Per-instance cross-check: GPU search (cyc_gpu.exists) vs brute force from the sum definition (verify156.exists_bf)
for EVERY n, cyclic k = 3..6; and interval thresholds vs OEIS A382397."""
import time
from cyc_gpu import exists
from verify156 import exists_bf
bad = []
for k in range(3, 7):
    t = time.time(); mx_g = mx_b = 0
    for n in range(k, (k**3 - k*k + 2*k)//2 + 2):
        g = exists(k, n, True, 3 if k > 3 else 2)[0]; b = exists_bf(k, n)
        if g != b: bad.append((k, n, g, b))
        if g: mx_g = n
        if b: mx_b = n
    print(f"cyclic k={k}: gpu max n = {mx_g}, brute force max n = {mx_b}, mismatches so far {len(bad)} [{time.time()-t:.0f}s]", flush=True)
print("MISMATCHES:", bad)
oeis = {3: 10, 4: 22, 5: 42, 6: 67}
for k in range(3, 7):
    mx = max(n for n in range(k, (k**3 - k*k + 2*k)//2 + 2) if exists(k, n, False, 3 if k > 3 else 2)[0])
    print(f"interval k={k}: gpu max n = {mx}, OEIS = {oeis[k]}", flush=True)
