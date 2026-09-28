import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
df=pd.read_csv("housing_regression.csv")
X=df[["Area"]]; y=df["Price"]
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=.2,random_state=42)
model=LinearRegression().fit(X_train,y_train)
pred=model.predict(X_test)
print("Simple Linear Regression implemented")
