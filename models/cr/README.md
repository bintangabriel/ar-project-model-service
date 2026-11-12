# CR (repossession) model slot

Put your CR model here and the service picks it up automatically on restart. Nothing else to change.

Required: `models/cr/reco.py` exposing

```python
def score(customer: dict, remo=None, min_cases: int = 0):
    """customer keys (same as the pay model): PEKERJAAN, SUMBER DANA, RINCIAN SUMBER DANA, ANSUS DETAIL,
    KEBERADAAN DEBITUR, KEBERADAAN UNIT, KATEGORI KASUS
    returns a pandas.DataFrame sorted best -> worst with columns:
        NAME_REMO        collector name
        P_CR             probability for the repossession outcome  (column name is set in registry.py -> GROUPS["cr"]["prob_col"])
        N_TRAIN_CASES    how many past cases that collector has in the CR training data
        REMO_OVERALL_PAY_RATE   (optional) the collector's overall success rate
    """
```

If your CR model was trained with other input fields, add them to the `Customer` model in `app.py` and to
`to_model_columns()`, and to the form. Until `models/cr/reco.py` exists, the API returns
`{"available": false}` for the CR group and the page shows "CR model not installed".
