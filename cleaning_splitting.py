import pandas as pd

da = pd.read_csv("bank-additional-full.csv", sep=";")
da = da.drop_duplicates()
da["y"] = (da["y"]=="yes").astype(int)
da["pdays_never"]=(da["pdays"]==999).astype(int)
da["pdays_days"]=da["pdays"].where(da["pdays"]!=999,0)

num=["age","campaign","pdays_days","pdays_never","previous","emp.var.rate","cons.price.idx","cons.conf.idx","euribor3m","nr.employed"]
cat=["job","marital","education","default","housing","loan","contact","month","day_of_week","poutcome"]
Xcols=num+cat

cut=int(len(da)*0.8)
train=da.iloc[:cut]
hold=da.iloc[cut:]

if __name__=="__main__":
    print(da.shape, train.shape, hold.shape)
    print("y all/train/hold", da["y"].mean(), train["y"].mean(), hold["y"].mean())
    print("cols", Xcols)
a