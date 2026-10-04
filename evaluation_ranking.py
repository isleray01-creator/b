from preprocessing_models import *
from cleaning_splitting import hold
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score

cmp=pd.DataFrame([
 {"set":"logreg","p500":pak(yho,probs["logreg"]),"lift500":lift(yho,probs["logreg"]),"prauc":scores(yho,probs["logreg"])["prauc"]},
 {"set":"logreg_bal","p500":pak(yho,probs["logreg_bal"]),"lift500":lift(yho,probs["logreg_bal"]),"prauc":scores(yho,probs["logreg_bal"])["prauc"]},
 {"set":"rf","p500":pak(yho,probs["rf"]),"lift500":lift(yho,probs["rf"]),"prauc":scores(yho,probs["rf"])["prauc"]},
 {"set":"gb","p500":pak(yho,probs["gb"]),"lift500":lift(yho,probs["gb"]),"prauc":scores(yho,probs["gb"])["prauc"]},
 {"set":"lda","p500":pak(yho,probs["lda"]),"lift500":lift(yho,probs["lda"]),"prauc":scores(yho,probs["lda"])["prauc"]},
 {"set":"rand","p500":rp,"lift500":rp/base,"prauc":base},
])

sc=table[table["model"]!="majority"].dropna(subset=["prauc"])
final=sc.sort_values(["lift500","prauc"],ascending=False).iloc[0]["model"]
fp=probs[final]
pred=(fp>=0.5).astype(int)
fn=(pred==0)&(yho==1)
fpp=(pred==1)&(yho==0)

top_o=np.argsort(fp)[::-1][:500]
top=hold.iloc[top_o].copy()
top.insert(0,"rank",range(1,501))
top["score"]=fp[top_o]
top["y_actual"]=yho[top_o]

if __name__=="__main__":
    print(cmp)
    print("final",final)
    print(scores(yho,fp))
    print("cm\n", confusion_matrix(yho,pred))
    for t in [0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5]:
        pr=(fp>=t).astype(int)
        print(t, precision_score(yho,pr,zero_division=0), recall_score(yho,pr,zero_division=0), f1_score(yho,pr,zero_division=0))
    print("fp",fpp.sum(),"fn",fn.sum())
    print("fn poutcome", hold.loc[fn,"poutcome"].value_counts().to_dict())
    print("fn success", (hold.loc[fn,"poutcome"]=="success").mean())
    print("fp success", (hold.loc[fpp,"poutcome"]=="success").mean())
    print(top[["rank","score","y_actual","age","job","education","contact","month","campaign","pdays","previous","poutcome"]].head(20))
    print("top500 y", top["y_actual"].mean(), "lift", top["y_actual"].mean()/base)
    print("base",base,"randp500",rp)
    print("features",Xcols)
    print("excluded duration")
    print("no csv")
