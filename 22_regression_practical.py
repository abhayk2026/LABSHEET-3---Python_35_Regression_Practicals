import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
df=pd.read_csv("housing_regression.csv"); X=df[["Area"]]; y=df.Price; m=make_pipeline(PolynomialFeatures(2),LinearRegression()).fit(X,y); print(m.predict(pd.DataFrame({"Area":[1500,2500]})))
