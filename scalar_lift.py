"""Test: Singer set A = points of the line V = F_q + F_q*theta in PG(2,q) = F_{q^3}^*/F_q^*.
Representatives v_a = 1 + a*theta (a in F_q), v_inf = theta.  For a projective class w (not on the line),
the ordered pairs (u,v) with v_u v_v = lambda * v_c * w (lambda in F_q^*, c a point) -- exactly q+1 of them.
Question: how is lambda distributed over F_q^* as (u,v) ranges over these pairs?  (q prime for simplicity)"""
import sys, itertools
from collections import Counter

def make_field(p):
    # find irreducible monic cubic x^3 - r2 x^2 - r1 x - r0 over F_p
    for r0 in range(1, p):
        for r1 in range(p):
            for r2 in range(p):
                if all((x**3 - r2*x*x - r1*x - r0) % p for x in range(p)):
                    return (r0, r1, r2)
    raise ValueError

def mul(a, b, p, r):
    r0, r1, r2 = r
    c = [0]*5
    for i in range(3):
        for j in range(3): c[i+j] = (c[i+j] + a[i]*b[j]) % p
    # reduce theta^4 = theta*theta^3
    # theta^3 = r0 + r1 t + r2 t^2 ; theta^4 = r0 t + r1 t^2 + r2 t^3 = r0 t + r1 t^2 + r2 (r0 + r1 t + r2 t^2)
    c4 = c[4]; c[1] = (c[1] + c4*r0) % p; c[2] = (c[2] + c4*r1) % p; c[3] = (c[3] + c4*r2) % p
    c3 = c[3]; c[0] = (c[0] + c3*r0) % p; c[1] = (c[1] + c3*r1) % p; c[2] = (c[2] + c3*r2) % p
    return (c[0], c[1], c[2])

def inv(a, p, r):
    # brute force via powering: a^(p^3-2)
    n = p**3 - 2; res = (1,0,0); base = a
    while n:
        if n & 1: res = mul(res, base, p, r)
        base = mul(base, base, p, r); n >>= 1
    return res

def main(p):
    r = make_field(p)
    pts = [(1, a, 0) for a in range(p)] + [(0, 1, 0)]     # v_a = 1 + a theta, v_inf = theta
    def on_line(x):
        """if x = lambda * v_c with lambda in F_p^*, return (lambda, c index) else None"""
        if x[2] != 0: return None
        if x[0] != 0:
            lam = x[0]; c = x[1] * pow(lam, -1, p) % p; return lam, c
        if x[1] != 0: return x[1], p   # lambda*theta
        return None
    # projective classes: enumerate all nonzero elements, group by scalar multiples
    seen = set(); results = {}
    for w in itertools.product(range(p), repeat=3):
        if w == (0,0,0) or w in seen: continue
        for s in range(1, p): seen.add(tuple(s*wi % p for wi in w))
        if on_line(w): continue            # w on the line => class in A
        winv = inv(w, p, r)
        lams = []
        for u in range(p+1):
            for v in range(p+1):
                x = mul(mul(pts[u], pts[v], p, r), winv, p, r)
                t = on_line(x)
                if t: lams.append(t[0])
        cnt = Counter(lams)
        results[w] = (len(lams), sorted(cnt.values(), reverse=True))
    shapes = Counter((n, tuple(v)) for n, v in results.values())
    print(f"p={p}: {len(results)} classes off the line; (number of ordered pairs, multiplicity pattern of lambda) -> count:")
    for k, c in shapes.most_common(): print("  ", k, c)

if __name__ == '__main__':
    for p in map(int, sys.argv[1:]): main(p)
