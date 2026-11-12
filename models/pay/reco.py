"""REMO recommender: predicts P(customer pays) for each candidate collector."""
import warnings; warnings.filterwarnings('ignore')
import pandas as pd, numpy as np, joblib, os
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
HERE = os.path.dirname(os.path.abspath(__file__))
CAT = ['PEKERJAAN','SUMBER DANA','RINCIAN SUMBER DANA','KEBERADAAN DEBITUR','KEBERADAAN UNIT','KATEGORI KASUS']

def load(path):
    """Reads the workbook (every sheet), cleans it, returns one row per customer-collector assignment."""
    parts=[]
    for name,df in pd.read_excel(path,sheet_name=None).items():
        if 'PEKERJAAN.1' in df.columns: df=df.drop(columns='PEKERJAAN').rename(columns={'PEKERJAAN.1':'PEKERJAAN'})
        t=[c for c in df.columns if c.lower().startswith('angke')][0]
        parts.append(df.rename(columns={t:'T'}).assign(SHEET=name))
    d=pd.concat(parts,ignore_index=True)
    d['T']=pd.to_numeric(d['T'],errors='coerce'); d['NO_AGGR']=d['NO_AGGR'].astype(str)
    for c in CAT+['NAME_COLLECTOR','ANSUS DETAIL']:
        d[c]=d[c].astype(str).str.strip().str.lower().replace({'nan':np.nan,'':np.nan})
    d=d[d['T']>=0].dropna(subset=['T','NAME_COLLECTOR']).copy()      # drop NAN/-1 results and rows with no collector
    d['y']=(d['T']>0).astype(int)
    # a customer's profile is frozen at its earliest row (what is known when the customer is first assigned);
    # later rows may contain notes written after a collector's visit, which a new customer would not have yet
    d=d.sort_values(['NO_AGGR','TGL_SK_KELUAR'])
    for c in CAT+['ANSUS DETAIL']: d[c]=d.groupby('NO_AGGR')[c].transform('first')
    return d.drop_duplicates(subset=['NO_AGGR','NAME_COLLECTOR','TGL_SK_KELUAR','y']).reset_index(drop=True)

def prep(d):
    X=d[CAT+['ANSUS DETAIL','NAME_COLLECTOR']].copy()
    for c in CAT: X[c]=X[c].fillna('unknown')
    X['ANSUS DETAIL']=X['ANSUS DETAIL'].fillna(''); X['NAME_COLLECTOR']=X['NAME_COLLECTOR'].fillna('unknown')
    return X

def _lr(C=0.2):
    tr=[('c',OneHotEncoder(handle_unknown='infrequent_if_exist',min_frequency=5),CAT),
        ('t',TfidfVectorizer(ngram_range=(1,2),min_df=3,sublinear_tf=True),'ANSUS DETAIL'),
        ('k',OneHotEncoder(handle_unknown='infrequent_if_exist',min_frequency=5),['NAME_COLLECTOR'])]
    return Pipeline([('t',ColumnTransformer(tr)),('m',LogisticRegression(C=C,max_iter=3000))])
def _hgb():
    cols=CAT+['NAME_COLLECTOR']
    enc=OrdinalEncoder(handle_unknown='use_encoded_value',unknown_value=np.nan,encoded_missing_value=np.nan,min_frequency=10)
    ct=ColumnTransformer([('o',enc,cols),('t',Pipeline([('v',TfidfVectorizer(ngram_range=(1,2),min_df=5,sublinear_tf=True)),('s',TruncatedSVD(20,random_state=0))]),'ANSUS DETAIL')])
    return Pipeline([('t',ct),('m',HistGradientBoostingClassifier(learning_rate=0.05,max_iter=150,max_depth=3,min_samples_leaf=30,
            l2_regularization=1.0,categorical_features=[True]*len(cols)+[False]*20,random_state=0))])
class Ens:
    """average of a logistic regression and a shallow gradient-boosting model"""
    def __init__(s): s.a=_lr(); s.b=_hgb()
    def fit(s,X,y): s.a.fit(X,y); s.b.fit(X,y); return s
    def predict_proba(s,X):
        p=(s.a.predict_proba(X)[:,1]+s.b.predict_proba(X)[:,1])/2; return np.c_[1-p,p]

def pairwise(df,s):
    """Within the same customer: how often the collector who got paid scores higher than a DIFFERENT collector who did not (0.5 = chance)."""
    t=df.assign(s=np.round(s,10)); w=n=0
    for _,h in t.groupby('NO_AGGR'):
        P=h[h.y==1]; N=h[h.y==0]
        for _,p in P.iterrows():
            for _,q in N.iterrows():
                if p.NAME_COLLECTOR!=q.NAME_COLLECTOR: n+=1; w+=(p.s>q.s)+0.5*(p.s==q.s)
    return w/n,n

def save(model,d): joblib.dump({'model':model,'counts':d.NAME_COLLECTOR.value_counts().to_dict(),'rate':d.groupby('NAME_COLLECTOR').y.mean().to_dict(),'base':float(d.y.mean())},os.path.join(HERE,'model.joblib'))

def score(customer, remo=None, min_cases=0):
    """customer: dict of PEKERJAAN, SUMBER DANA, RINCIAN SUMBER DANA, ANSUS DETAIL, KEBERADAAN DEBITUR, KEBERADAAN UNIT, KATEGORI KASUS.
    remo: a name, a list of names, or None for every REMO in the training data."""
    art=joblib.load(os.path.join(HERE,'model.joblib'))
    names=[remo] if isinstance(remo,str) else (list(remo) if remo else list(art['counts']))
    rows=[{**{k:str(v).strip().lower() for k,v in customer.items()},'NAME_COLLECTOR':str(n).strip().lower()} for n in names]
    X=prep(pd.DataFrame(rows)); p=art['model'].predict_proba(X)[:,1]
    out=pd.DataFrame({'NAME_REMO':[str(n).lower() for n in names],'P_PAY':p.round(3),
                      'N_TRAIN_CASES':[art['counts'].get(str(n).lower(),0) for n in names],
                      'REMO_OVERALL_PAY_RATE':[round(art['rate'].get(str(n).lower(),np.nan),3) for n in names]})
    out=out[out.N_TRAIN_CASES>=min_cases]
    return out.sort_values('P_PAY',ascending=False).reset_index(drop=True)
