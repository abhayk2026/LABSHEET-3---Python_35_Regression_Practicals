import pandas as pd
df=pd.read_csv("housing_regression.csv")
X=df[["Area","Bedrooms","Age"]]; y=df["Price"]
print("Independent:",X.columns.tolist()); print("Dependent:",y.name)
