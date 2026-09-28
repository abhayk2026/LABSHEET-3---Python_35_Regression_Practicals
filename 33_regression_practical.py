from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
d=load_diabetes(as_frame=True); X=d.data; y=d.target; a,b,c,e=train_test_split(X,y,test_size=.2,random_state=42); m=LinearRegression().fit(a,c); print("Diabetes R2:",r2_score(e,m.predict(b)))
