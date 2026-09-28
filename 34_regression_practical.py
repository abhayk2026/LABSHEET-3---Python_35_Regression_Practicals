import pandas as pd
import joblib
from sklearn.linear_model import LinearRegression
df=pd.read_csv("housing_regression.csv"); m=LinearRegression().fit(df[["Area","Bedrooms","Age"]],df.Price); joblib.dump(m,"regression_model.joblib"); print("Saved regression_model.joblib")
