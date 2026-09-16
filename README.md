# Erdős problem #156 — small maximal Sidon sets: exact small-case data and a failed construction idea

**Problem.** Does there exist a maximal Sidon set A ⊂ {1, …, N} of size O(N^{1/3})?
(https://www.erdosproblems.com/156). Known: every maximal Sidon set has size ≫ N^{1/3}; Ruzsa (1998)
constructed one of size ≪ (N log N)^{1/3}. The same bound, with the log factor, is the best known in Z_2^n
(Redman–Rose–Walker, SIAM J. Discrete Math. 2022, arXiv:2109.00292).

**Outcome.** Not resolved. This repository records (1) exact thresholds for the cyclic-group analogue of the
problem for k ≤ 7, computed on a GPU and cross-checked by a brute-force program written directly from the
definitions, and (2) why one natural algebraic attempt to remove the log factor behaves like a random choice.

## 1. Exact thresholds

Definitions. A ⊂ G (G = Z_n, or the integers) is Sidon if a + b = c + d with a, b, c, d ∈ A implies
{a, b} = {c, d} (repeated elements allowed). A Sidon set is maximal if no element of G can be added. In Z_n
with n even this definition blocks every y = a + n/2 (since y + y = a + a); for odd n and for intervals the
distinction does not arise.

Let c(k) be the largest n such that Z_n contains a maximal Sidon set of size k, and i(k) the same for the
interval {1, …, n} (OEIS A382397; i(8) = 144 is proved in F. Huber's 2026 preprint linked there).

| k | i(k) interval | i(k)/k³ | c(k) cyclic | c(k)/k³ | witness in Z_{c(k)} |
|---|---|---|---|---|---|
| 3 | 10 | 0.37 | 12 | 0.44 | {0, 1, 4} |
| 4 | 22 | 0.34 | 28 | 0.44 | {0, 18, 25, 26} |
| 5 | 42 | 0.34 | 50 | 0.40 | {0, 7, 19, 20, 22} |
| 6 | 67 | 0.31 | 84 | 0.39 | {0, 7, 55, 56, 58, 67} |
| 7 | 101 | 0.29 | 126 | 0.37 | {0, 21, 50, 103, 104, 119, 121} |
| 8 | 144 | 0.28 | | | |

For k = 7, maximal 7-sets exist in Z_n for every 48 ≤ n ≤ 118 and then only for n = 120, 122, 124, 126.
The trivial bound is c(k) ≤ (k³ − k² + 2k)/2. The extremal sets consist of a small dense cluster plus a few far
points; no algebraic structure is visible, and the ratio n/k³ decreases slowly in both settings.

Verification. `cyc_gpu.py` (CuPy + numba: the CPU enumerates Sidon prefixes, each GPU thread completes one
prefix and tests maximality) was compared instance by instance with `verify156.exists_bf`, an independent
brute force over all k-subsets containing 0 using the sum definition, for every n at k = 3..6 (`crosscheck.py`,
no mismatch), and for k = 7 on 121 ≤ n ≤ 135 (`bf_check7.log`: solutions exactly at 122, 124, 126). All 75
witnesses printed for k = 7 were re-checked from the definition (`verify_witnesses.py`). The interval version
of the program reproduces i(3..6) = 10, 22, 42, 67 from A382397.

## 2. Why the natural algebraic lift does not remove the log

A Sidon set A ⊂ [1, N] is maximal iff every x ∉ A is of the form a + b − c (a, b, c ∈ A) or (a + b)/2.
With |A| = K there are about K³/2 values a + b − c, so K ≥ (2N)^{1/3}(1 + o(1)), and a construction of size
O(N^{1/3}) has to cover [1, N] with these values almost without overlap. Ruzsa lifts a Singer set from Z_M
(M = q² + q + 1) to [1, N] by choosing one of L ≈ N/M lifts per element at random; the union bound over the
N points costs the factor (log N)^{1/3}.

The Singer set is the point set of PG(2, q), i.e. the lines through 0 in F_{q³}, and each point has q − 1 scalar
representatives in F_{q³}^*, a natural deterministic lift. `scalar_lift.py` computes, for each target class,
the multiset of scalars produced by the (q + 1)/2 triples hitting it. For q = 5, 7, 11, 13 these scalars take
only about half of the q − 1 possible values, with collisions, i.e. the algebraic lift behaves like a random
one. Weil-type equidistribution controls only classes of size ≥ √q, which would give exponent 2/5, not 1/3.

## Files

`cyc_gpu.py`, `cyc_search.py` (CPU-only predecessor), `verify156.py`, `crosscheck.py`, `bf_check.py`,
`verify_witnesses.py`, `scalar_lift.py`; logs `cyc7.log`, `fixed_runs2.log`, `bf_check7.log`, `verify.log`.
