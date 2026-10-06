from cleaning_splitting import da, train, hold, Xcols, num, cat
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score

pre=ColumnTransformer([
 ("num",Pipeline([("imp",SimpleImputer(strategy="median")),("sc",StandardScaler())]),num),
 ("cat",Pipeline([("imp",SimpleImputer(strategy="most_frequent")),("oh",OneHotEncoder(handle_unknown="ignore",sparse_output=False))]),cat),
])

def pp(m):
    return Pipeline([("pre",pre),("model",m)])

def pak(y,p,k=500):
    o=np.argsort(p)[::-1][:k]
    return y[o].mean()

def lift(y,p,k=500):
    b=y.mean()
    return pak(y,p,k)/b if b else np.nan

def rndp(y,k=500):
    rng=np.random.default_rng(42)
    return y[rng.permutation(len(y))[:k]].mean()

def scores(y,p):
    pr=(p>=0.5).astype(int)
    return {
        "acc":accuracy_score(y,pr),
        "prec":precision_score(y,pr,zero_division=0),
        "rec":recall_score(y,pr,zero_division=0),
        "f1":f1_score(y,pr,zero_division=0),
        "roc":roc_auc_score(y,p),
        "prauc":average_precision_score(y,p),
        "p500":pak(y,p),
        "lift500":lift(y,p),
    }

ytr=train["y"].values
yho=hold["y"].values
Xtr=train[Xcols]
Xho=hold[Xcols]
base=yho.mean()
rp=rndp(yho)

mods={
 "majority":None,
 "logreg":LogisticRegression(max_iter=5001),
 "logreg_bal":LogisticRegression(max_iter=5001,class_weight="balanced"),
 "rf":RandomForestClassifier(n_estimators=200,random_state=42,class_weight="balanced",n_jobs=-1),
 "gb":GradientBoostingClassifier(random_state=42),
 "lda":LinearDiscriminantAnalysis(),
 "qda":QuadraticDiscriminantAnalysis(),
}

rows=[]
probs={}
for name,m in mods.items():
    if name=="majority":
        const=np.full(len(yho),ytr.mean())
        s=scores(yho,const)
        s["p500"]=rp
        s["lift500"]=rp/base
        rows.append({"model":name,**s})
        probs[name]=const
        continue
    try:
        pipe=pp(m)
        pipe.fit(Xtr,ytr)
        p=pipe.predict_proba(Xho)[:,1]
        rows.append({"model":name,**scores(yho,p)})
        probs[name]=p
        print(name,"ok")
    except Exception as e:
        rows.append({"model":name,"err":str(e)[:70]})
        print(name,"fail",str(e)[:70])

table=pd.DataFrame(rows)

if __name__=="__main__":
    print("base",base,"rand",rp,"maj",(yho==0).mean())
    print(table)
    print("chosen check later lift500")
    print(table.dropna(subset=["prauc"]).sort_values(["lift500","prauc"],ascending=False)[["model","acc","prec","rec","f1","roc","prauc","p500","lift500"]])
