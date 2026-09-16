import time, sys
from verify156 import exists_bf
k = int(sys.argv[1]); lo = int(sys.argv[2]); hi = int(sys.argv[3])
t = time.time(); res = [n for n in range(lo, hi + 1) if exists_bf(k, n)]
print(f"brute force k={k}, n in [{lo},{hi}]: exists for n = {res}  [{time.time()-t:.0f}s]", flush=True)
