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



top_o=np.argsort(fp)[::-1][:500]
top=hold.iloc[top_o].copy()
top.insert(0,"rank",range(1,501))
top["score"]=fp[top_o]
top["y_actual"]=yho[top_o]

if __name__=="__main__":
    print(cmp)
    print("final",final)
    print(scores(yho,fp))
    print(top[["rank","score","y_actual","age","job","education","contact","month","campaign","pdays","previous","poutcome"]].head(20))
    print("top500 y", top["y_actual"].mean(), "lift", top["y_actual"].mean()/base)


