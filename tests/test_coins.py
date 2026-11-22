"""Coins: the default threshold of an intensity model is found on the tape and integrated exactly.
Checks on the credit-linked autocallable of benchmarks/bench_cox_coins.py (2000 paths):
  - the coin is found without a declaration (one input, 104 comparisons, exponential law);
  - the fast path (CoxDefault markup) equals the two-pass coin term to 1e-10;
  - spot Greeks are identical with and without coins (no coin in them);
  - the default Greeks agree with the delta correction on the default directions within 1% on the same paths."""
import os, sys, io, contextlib
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..")); sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "benchmarks"))
import numpy as np
import bench_cox_coins as B

plain, marked = B.record(False), B.record(True)
Z = np.random.default_rng(5).standard_normal((2000, len(plain["za"])))
res, d_auto = {}, None
for name, t, co in (("off", plain, "off"), ("coin", plain, "auto"), ("fast", marked, "auto")):
    d = B.driver(t, co)
    with contextlib.redirect_stdout(io.StringIO()):
        res[name] = np.asarray(d.run(Z)["total"])
    if name == "coin":
        d_auto = d
diag = d_auto.coins.diagnostics
coin = [k for k in diag if k != "candidates"]
assert len(coin) == 1 and diag[coin[0]]["comparisons"] == 104 and diag[coin[0]]["law"] == "exp", diag
assert np.allclose(res["fast"], res["coin"], rtol=1e-10, atol=1e-12), (res["fast"], res["coin"])
assert np.allclose(res["coin"][:2], res["off"][:2], rtol=1e-12, atol=1e-14), (res["coin"], res["off"])
assert np.all(np.abs(res["coin"][2:] - res["off"][2:]) <= 0.01 * np.abs(res["off"][2:]) + 0.02), (res["coin"], res["off"])
print("coins:", diag)
print("off ", np.round(res["off"], 5)); print("coin", np.round(res["coin"], 5)); print("fast", np.round(res["fast"], 5))
print("\nAll coin tests passed.")
