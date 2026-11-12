"""CR predictor: probability that each collector (remo) successfully repossesses a unit.

Requires cr_model.pkl in the same folder.

    from cr_predictor import predict_cr

    result = predict_cr({
        "PEKERJAAN": "WIRASWASTA",
        "SUMBER DANA": "USAHA LAIN/UMKM",
        "KEBERADAAN DEBITUR": "MASIH DI ALAMAT DOMISILI",
        "KEBERADAAN UNIT": "UNIT ADA DI DEBITUR",
        "KATEGORI KASUS": "BAD CHARACTER",
    }, top_n=10)

Run `python cr_predictor.py` for a demo. The function is pure (no I/O after the model is
loaded at import), so it can be dropped straight into an endpoint handler.
"""
import os
import re
import pickle
from typing import Optional

import numpy as np
import pandas as pd
from scipy import sparse

CUSTOMER_FIELDS = ["PEKERJAAN", "SUMBER DANA", "KEBERADAAN DEBITUR", "KEBERADAAN UNIT", "KATEGORI KASUS"]

_art = pickle.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "cr_model.pkl"), "rb"))
_model, _oh_c, _oh_k, _KS = _art["model"], _art["oh_c"], _art["oh_k"], _art["KS"]
_collectors = list(_art["eligible"])  # collectors with >=30 historical SK, best first
_levels = dict(zip(CUSTOMER_FIELDS, _oh_c.categories_))
_MISSING = {"", "NAN", "NONE", "NULL", "0"}


def _clean(value) -> str:
    """Same normalisation used in training."""
    if value is None:
        return "TIDAK DIISI"
    s = re.sub(r"\s+", " ", re.sub(r"\s*/\s*", "/", str(value).strip().upper()))
    return "TIDAK DIISI" if s in _MISSING else s


def predict_cr(customer: dict, top_n: Optional[int] = None) -> dict:
    """Return the CR probability of every ranked collector for one customer.

    customer: dict with the 5 keys in CUSTOMER_FIELDS (missing/empty value is allowed and
              treated as 'not filled'). Unknown or rare values fall back to 'LAINNYA'.
    top_n:    keep only the best N collectors (default: all ranked collectors).
    """
    absent = [k for k in CUSTOMER_FIELDS if k not in customer]
    if absent:
        raise ValueError(f"Missing customer fields: {absent}")

    c = {k: _clean(customer[k]) for k in CUSTOMER_FIELDS}
    if c["PEKERJAAN"] == "TIDAK BEKERJA":
        c["PEKERJAAN"] = "TIDAK ADA PEKERJAAN"
    for k in CUSTOMER_FIELDS:
        if c[k] not in _levels[k] and "LAINNYA" in _levels[k]:
            c[k] = "LAINNYA"

    n = len(_collectors)
    cust = _oh_c.transform(pd.DataFrame([c] * n, columns=CUSTOMER_FIELDS))
    coll = _oh_k.transform(pd.DataFrame({"NAME_COLLECTOR": _collectors})) * _KS
    probs = _model.predict_proba(sparse.hstack([cust, coll]).tocsr())[:, 1]

    order = np.argsort(-probs)
    if top_n:
        order = order[:top_n]
    return {
        "customer": c,
        "n_collectors_ranked": n,
        "ranking": [
            {"rank": i + 1, "collector": _collectors[j], "probability_cr": round(float(probs[j]), 4)}
            for i, j in enumerate(order)
        ],
    }


if __name__ == "__main__":
    import json

    demo = {
        "PEKERJAAN": "WIRASWASTA",
        "SUMBER DANA": "USAHA LAIN/UMKM",
        "KEBERADAAN DEBITUR": "MASIH DI ALAMAT DOMISILI",
        "KEBERADAAN UNIT": "UNIT ADA DI DEBITUR",
        "KATEGORI KASUS": "BAD CHARACTER",
    }
    print(json.dumps(predict_cr(demo, top_n=5), indent=2, ensure_ascii=False))
