"""Model registry: one entry per recommendation group.

Each group lives in models/<group>/ and must contain a `reco.py` that exposes

    score(customer: dict, remo=None, min_cases: int = 0) -> pandas.DataFrame

with at least these columns:  NAME_REMO, <prob_col>, N_TRAIN_CASES   (REMO_OVERALL_PAY_RATE is optional)
sorted from best to worst. `models/pay/reco.py` is your payment model, unchanged.
To plug in the CR (repossession) model: drop its files into models/cr/ (see models/cr/README.md).
"""
import importlib.util, os, sys, threading
import joblib

BASE = os.path.dirname(os.path.abspath(__file__))

GROUPS = {
    "pay": {"dir": "models/pay", "prob_col": "P_PAY", "label": "Payment"},
    "cr":  {"dir": "models/cr",  "prob_col": "P_CR",  "label": "CR (repossession)"},
}

# --- load each .joblib only once -------------------------------------------------------------
# reco.score() calls joblib.load() on every request (~2 MB, unpickling takes tens of ms). Cache it by path.
_orig_load, _cache, _lock = joblib.load, {}, threading.Lock()
def _cached_load(path, *a, **k):
    key = os.path.abspath(str(path))
    with _lock:
        if key not in _cache:
            _cache[key] = _orig_load(path, *a, **k)
        return _cache[key]
joblib.load = _cached_load
# ----------------------------------------------------------------------------------------------

class Group:
    def __init__(self, name, cfg):
        self.name, self.prob_col, self.label = name, cfg["prob_col"], cfg["label"]
        self.dir = os.path.join(BASE, cfg["dir"])
        self.module, self.error = None, None
        self._load()

    def _load(self):
        path = os.path.join(self.dir, "reco.py")
        if not os.path.isfile(path):
            self.error = f"no model installed (missing {os.path.relpath(path, BASE)})"; return
        try:
            sys.path.insert(0, self.dir)                      # lets reco.py import sibling files
            spec = importlib.util.spec_from_file_location(f"model_{self.name}", path)
            mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
            if not hasattr(mod, "score"): raise AttributeError("reco.py has no score() function")
            self.module = mod
            mod.score(_WARMUP, None, 10**9)                   # warm up + fail fast at startup
        except Exception as e:                                # a broken CR model must not take the pay model down
            self.module, self.error = None, f"failed to load: {type(e).__name__}: {e}"
        finally:
            if sys.path and sys.path[0] == self.dir: sys.path.pop(0)

    @property
    def available(self): return self.module is not None

    def recommend(self, customer, top_n, min_cases):
        df = self.module.score(customer, None, min_cases).head(top_n)
        items = []
        for i, r in enumerate(df.itertuples(index=False), 1):
            r = r._asdict()
            rate = r.get("REMO_OVERALL_PAY_RATE")
            items.append({
                "rank": i,
                "peojf": str(r["NAME_REMO"]).upper(),
                "probability": float(r[self.prob_col]),
                "nCases": int(r["N_TRAIN_CASES"]),
                "overallRate": None if rate is None or rate != rate else float(rate),
            })
        return items

_WARMUP = {"PEKERJAAN": "wiraswasta", "SUMBER DANA": "lain-lain", "RINCIAN SUMBER DANA": "lain-lain", "ANSUS DETAIL": "",
           "KEBERADAAN DEBITUR": "masih di alamat domisili", "KEBERADAAN UNIT": "tidak diketahui", "KATEGORI KASUS": "bad character"}

def load_registry():
    return {name: Group(name, cfg) for name, cfg in GROUPS.items()}
