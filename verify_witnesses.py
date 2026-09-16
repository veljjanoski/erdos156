import re
from verify156 import maximal
n_ok = 0
for line in open('cyc7.log'):
    m = re.search(r"n=(\d+): maximal=True.*witness=\[([\d, ]+)\]", line)
    if m:
        n = int(m.group(1)); B = [int(x) for x in m.group(2).split(',')]
        assert maximal(B, n), (n, B); n_ok += 1
print(f"{n_ok} k=7 witnesses verified from the sum definition")
