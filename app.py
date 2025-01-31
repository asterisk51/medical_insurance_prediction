import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression 
from sklearn.metrics import r2_score 
import streamlit as st

medical_df = pd.read_csv('insurance.csv')

medical_df.head()

medical_df.info()

medical_df.describe()

medical_df.replace({'sex':{'male':0,'female':1}},inplace=True)
medical_df.replace({'smoker':{'yes':0,'no':1}},inplace=True)
medical_df.replace({'region':{'southeast':0,'southwest':1,'northwest':2,'northeast':3}},inplace=True)

X = medical_df.drop('charges',axis=1)
y = medical_df['charges']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=2)

lg = LinearRegression()
lg.fit(X_train,y_train) # 80% model will be train
y_pred = lg.predict(X_test) # 10% model will be predicted 

r2_score(y_test,y_pred)

#web_app
st.title("Medical Insurance prediction Model")
input_text = st.text_input("Enter person's all features ")
input_text_splited = input_text.split(",")
try: 
    np_df = np.asarray(input_text_splited,dtype=float)
    prediction = lg.predict(np_df.reshape(1,-1))
    st.write("Medical Insurance for this person is :\n", prediction[0])
except ValueError:
    st.write("Please enter numerical value")

 
