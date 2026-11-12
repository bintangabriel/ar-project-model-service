"""Builds options.json (dropdown values for the form) from the training workbook.
Usage: python make_options.py "Data_Train.xlsx"      (run again whenever you retrain)"""
import sys, json, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "pay"))
from reco import load

FIELDS = {"pekerjaan": "PEKERJAAN", "sumberDana": "SUMBER DANA", "keberadaanDebitur": "KEBERADAAN DEBITUR",
          "keberadaanUnit": "KEBERADAAN UNIT", "kategoriKasus": "KATEGORI KASUS"}
MIN_COUNT = 10   # ignore typos / one-off values

d = load(sys.argv[1]).drop_duplicates("NO_AGGR")      # one row per customer
out = {}
for key, col in FIELDS.items():
    vc = d[col].dropna().value_counts()
    out[key] = [v for v, n in vc.items() if n >= MIN_COUNT]
json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "options.json"), "w"), indent=2, ensure_ascii=False)
for k, v in out.items(): print(k, len(v), v[:6])
