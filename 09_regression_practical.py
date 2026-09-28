import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
df=pd.read_csv("housing_regression.csv"); X=df[["Area"]]; y=df["Price"]
m=LinearRegression().fit(X,y); plt.scatter(X,y); plt.plot(X,m.predict(X)); plt.xlabel("Area"); plt.ylabel("Price"); plt.title("Linear Regression"); plt.show()
