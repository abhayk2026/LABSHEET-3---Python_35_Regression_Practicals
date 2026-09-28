import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
df=pd.read_csv("housing_regression.csv"); X=df[["Area"]]; y=df.Price; plt.scatter(X,y); g=np.linspace(X.min(),X.max(),200).reshape(-1,1)
for d in [2,3]: plt.plot(g,make_pipeline(PolynomialFeatures(d),LinearRegression()).fit(X,y).predict(g),label=f"Degree {d}")
plt.legend(); plt.show()
