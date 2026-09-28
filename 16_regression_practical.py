import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
df=pd.read_csv("housing_regression.csv"); X=df[["Area","Bedrooms","Age"]]; y=df.Price; a,b,c,d=train_test_split(X,y,test_size=.2,random_state=42); m=LinearRegression().fit(a,c); p=m.predict(b); plt.plot(d.values,label="Actual"); plt.plot(p,label="Predicted"); plt.legend(); plt.show()
