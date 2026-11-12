"""Internal ML service. Run:  uvicorn app:app --host 127.0.0.1 --port 8000
Only the Express backend should call this (keep it off the public internet)."""
import json, os, secrets
from typing import Optional
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, Field
from registry import load_registry, GROUPS

SERVICE_KEY = os.environ.get("ML_SERVICE_KEY", "")          # shared secret with Express; empty = no check (local dev only)
REG = load_registry()
OPTIONS = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "options.json"), encoding="utf-8"))
app = FastAPI(title="PEOJF recommendation service")

class Customer(BaseModel):
    pekerjaan: str = Field(min_length=1, max_length=100)
    sumberDana: str = Field(min_length=1, max_length=100)
    rincianSumberDana: str = Field(default="", max_length=300)
    ansusDetail: str = Field(default="", max_length=2000)
    keberadaanDebitur: str = Field(min_length=1, max_length=100)
    keberadaanUnit: str = Field(min_length=1, max_length=100)
    kategoriKasus: str = Field(min_length=1, max_length=100)

class RecommendRequest(BaseModel):
    customer: Customer
    topN: int = Field(default=5, ge=1, le=20)
    minCases: int = Field(default=20, ge=0, le=1000)

def to_model_columns(c: Customer) -> dict:
    """camelCase API fields -> the column names the models were trained with"""
    return {"PEKERJAAN": c.pekerjaan, "SUMBER DANA": c.sumberDana, "RINCIAN SUMBER DANA": c.rincianSumberDana,
            "ANSUS DETAIL": c.ansusDetail, "KEBERADAAN DEBITUR": c.keberadaanDebitur,
            "KEBERADAAN UNIT": c.keberadaanUnit, "KATEGORI KASUS": c.kategoriKasus}

def check_key(key: Optional[str]):
    if SERVICE_KEY and not secrets.compare_digest(key or "", SERVICE_KEY):
        raise HTTPException(status_code=401, detail="bad service key")

@app.get("/health")
def health():
    return {"status": "ok", "groups": {n: {"available": g.available, "reason": g.error} for n, g in REG.items()}}

@app.get("/options")
def options(x_service_key: Optional[str] = Header(default=None)):
    check_key(x_service_key); return OPTIONS

@app.post("/recommend")
def recommend(req: RecommendRequest, x_service_key: Optional[str] = Header(default=None)):
    check_key(x_service_key)
    cust = to_model_columns(req.customer)
    results = {}
    for name, g in REG.items():                                # run `def` (not async) so FastAPI uses a thread pool
        if not g.available:
            results[name] = {"available": False, "label": g.label, "reason": g.error, "items": []}; continue
        try:
            results[name] = {"available": True, "label": g.label, "items": g.recommend(cust, req.topN, req.minCases)}
        except Exception as e:                                 # one failing model should not hide the other
            results[name] = {"available": False, "label": g.label, "reason": f"{type(e).__name__}: {e}", "items": []}
    return {"results": results, "meta": {"topN": req.topN, "minCases": req.minCases}}
