import pandas as pd
import joblib
m=joblib.load("regression_model.joblib"); new=pd.DataFrame({"Area":[1500,2500],"Bedrooms":[3,5],"Age":[5,2]}); print(m.predict(new))
