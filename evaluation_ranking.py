from preprocessing_models import *
from cleaning_splitting import hold
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score



sc=table[table["model"]!="majority"].dropna(subset=["prauc"])
final=sc.sort_values(["lift500","prauc"],ascending=False).iloc[0]["model"]
fp=probs[final]



top_o=np.argsort(fp)[::-1][:500]
top=hold.iloc[top_o].copy()
top.insert(0,"rank",range(1,501))
top["score"]=fp[top_o]
top["y_actual"]=yho[top_o]

if __name__=="__main__":

    print("final",final)
    print(scores(yho,fp))
    print(top[["rank","score","y_actual","age","job","education","contact","month","campaign","pdays","previous","poutcome"]].head(20))
    print("top500 y", top["y_actual"].mean(), "lift", top["y_actual"].mean()/base)


