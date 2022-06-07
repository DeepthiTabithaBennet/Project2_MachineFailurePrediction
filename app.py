# !pip install streamlit

import pandas as pd
import numpy as np
# import seaborn as sns
# import matplotlib.pyplot as plt
# %matplotlib inline
import io
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
import streamlit as st


st.title("Failure Prediction App")


data = pd.read_csv('predictive_maintenance.csv')


data = data.drop(["UDI",'Product ID'],axis=1)


from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()

label_encoder.fit(data['Type'])
data['Type'] = label_encoder.transform(data['Type'])

label_encoder.fit(data['Target'])
data['Target'] = label_encoder.transform(data['Target'])


from sklearn.model_selection import train_test_split


X_train, X_test, y_train, y_test = train_test_split(data.drop(['Failure Type','Target'],axis=1),
                                                    data['Target'], test_size=0.3, random_state=42)


# Logistic Regression
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()


lr.fit(X_train, y_train)
print(lr.score(X_test, y_test))


# Type = (st.number_input("Enter the Type of Machine"))
TypeInput = st.selectbox("Select the Machine Type", ['M', 'L', 'H'])


AirTemperature = (st.number_input("Enter the Air Temperature"))

ProcessTemperature = (st.number_input("Enter the Process Temperature"))

RotationalSpeed = (st.number_input("Enter the Rotational Speed"))

Torque = (st.number_input("Enter the Torque"))

ToolWear = (st.number_input("Enter the Tool Wear"))


if (TypeInput == 'M'):
  Type = 2

elif (TypeInput == 'L'):
  Type = 1

elif (TypeInput == 'H'):
  Type = 0
  

outcome = lr.predict([[Type, AirTemperature, ProcessTemperature, RotationalSpeed, Torque, ToolWear]])


number = preprocessing.LabelEncoder()

data["Type"] = number.fit_transform(data["Type"])
data["Failure Type"] = number.fit_transform(data["Failure Type"])


X = data.drop(["Failure Type"], axis = 1)
y = data["Failure Type"]

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 0, test_size = 0.25)


scaler = StandardScaler()
scaler.fit(X_train)

X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)


model = LogisticRegression(multi_class = "multinomial", solver = "saga")
model.fit(X_train, y_train)

print(model.score(X_test, y_test))


OutcomeType = model.predict([[Type, AirTemperature, ProcessTemperature, RotationalSpeed, Torque, ToolWear, outcome[0]]])


if (st.button("Submit")):

  st.header("Prediction : ")

  if (outcome[0] == 0):
    st.success("No Failure")
  if (outcome[0] == 1):
    st.error("Failure")

  st.header("If any Failure occurs, chances are, the reason will be : ")

  if (OutcomeType[0] == 0):
    st.info("Heat Dissipation Failure")

  elif (OutcomeType[0] == 1 or outcome[0] == 0):
    st.info("No Failure")

  elif (OutcomeType[0] == 2):
    st.info("Overstrain Failure")

  elif (OutcomeType[0] == 3):
    st.info("Power Failure")

  elif (OutcomeType[0] == 4):
    st.info("Random Failure")

  elif (OutcomeType[0] == 5):
    st.info("Tool Wear Failure")
