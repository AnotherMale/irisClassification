import streamlit as st
import joblib
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris

@st.cache_resource
def load_model():
    try:
        model = joblib.load('iris_model.pkl')
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

iris = load_iris()
target_names = iris.target_names
feature_names = iris.feature_names

st.set_page_config(page_title="Iris Flower Classifier", page_icon="🌸")

st.title("Iris Flower Classification")
st.write("""
This app predicts the species of an Iris flower based on its measurements.
""")

st.sidebar.header("Input Features")

def get_user_input():
    sepal_length = st.sidebar.slider('Sepal length (cm)', 4.0, 8.0, 5.1)
    sepal_width = st.sidebar.slider('Sepal width (cm)', 2.0, 5.0, 3.5)
    petal_length = st.sidebar.slider('Petal length (cm)', 1.0, 7.0, 1.4)
    petal_width = st.sidebar.slider('Petal width (cm)', 0.1, 2.5, 0.2)
    
    user_data = {
        'sepal_length': sepal_length,
        'sepal_width': sepal_width,
        'petal_length': petal_length,
        'petal_width': petal_width
    }
    
    features = pd.DataFrame(user_data, index=[0])
    return features

user_input = get_user_input()

st.subheader("User Input Features")
st.write(user_input)

model = load_model()
if model is not None:
    prediction = model.predict(user_input)
    prediction_proba = model.predict_proba(user_input)
    
    st.subheader("Prediction")
    st.write(f"The predicted species is **{target_names[prediction[0]]}**")
    
    st.subheader("Prediction Probability")
    proba_df = pd.DataFrame(prediction_proba, columns=target_names)
    st.write(proba_df)
    
    st.bar_chart(proba_df.T)
else:
    st.error("Model could not be loaded. Please check the model file.")

st.sidebar.markdown("""
### About the Dataset
The Iris dataset contains measurements for 150 iris flowers from three species:
- Setosa
- Versicolor
- Virginica

The measurements are:
- Sepal length (cm)
- Sepal width (cm)
- Petal length (cm)
- Petal width (cm)
""")
