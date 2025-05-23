import streamlit as st
import joblib
import numpy as np
import pandas as pd

model = joblib.load('iris_model.pkl')

feature_names = iris.feature_names

st.title("Iris Flower Classifier")
st.write("Enter measurements for the Iris flower:")

sepal_length = st.slider("Sepal Length (cm)", float(iris.data[:,0].min()), float(iris.data[:,0].max()), float(iris.data[:,0].mean()))
sepal_width = st.slider("Sepal Width (cm)", float(iris.data[:,1].min()), float(iris.data[:,1].max()), float(iris.data[:,1].mean()))
petal_length = st.slider("Petal Length (cm)", float(iris.data[:,2].min()), float(iris.data[:,2].max()), float(iris.data[:,2].mean()))
petal_width = st.slider("Petal Width (cm)", float(iris.data[:,3].min()), float(iris.data[:,3].max()), float(iris.data[:,3].mean()))

input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])

if st.button('Predict'):
    prediction = model.predict(input_data)
    predicted_species = iris.target_names[prediction][0]
    st.write(f"**Predicted species:** {predicted_species}")