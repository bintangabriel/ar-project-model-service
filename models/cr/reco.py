"""Adapter: makes cr_predictor.py look like the other models (score(...) -> table), so registry.py can use it.
Do not put logic here; the CR model itself lives in cr_predictor.py + cr_model.pkl."""
import pandas as pd
from cr_predictor import predict_cr, CUSTOMER_FIELDS   # loads cr_model.pkl once, when the service starts

# cr_model.pkl only stores collectors with >= 30 historical SK, not exact counts.
# We report 30 and registry.py marks it as "30+" (GROUPS["cr"]["ncases_is_min"]).
MIN_HISTORICAL_SK = 30

def score(customer, remo=None, min_cases=0):
    """customer: the service's dict (7 fields). CR uses only the 5 in CUSTOMER_FIELDS; the other keys are ignored."""
    res = predict_cr({k: customer.get(k) for k in CUSTOMER_FIELDS})        # all ranked collectors, best first
    df = pd.DataFrame({
        "NAME_REMO": [r["collector"] for r in res["ranking"]],
        "P_CR": [r["probability_cr"] for r in res["ranking"]],
        "N_TRAIN_CASES": MIN_HISTORICAL_SK,
    })
    if remo is not None:
        wanted = {str(x).strip().upper() for x in ([remo] if isinstance(remo, str) else remo)}
        df = df[df.NAME_REMO.astype(str).str.upper().isin(wanted)]
    return df[df.N_TRAIN_CASES >= min_cases].reset_index(drop=True)
