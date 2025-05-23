import streamlit as st
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from sklearn.datasets import load_iris

sns.set_style("whitegrid")

@st.cache_resource
def load_model():
    try:
        model = joblib.load('iris_model.pkl')
        from sklearn import __version__ as skv
        st.sidebar.info(f"scikit-learn version: {skv}")
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.error("Please ensure the model was trained with a compatible scikit-learn version.")
        return None

iris = load_iris()
target_names = iris.target_names
feature_names = iris.feature_names

iris_df = pd.DataFrame(iris.data, columns=feature_names)
iris_df['species'] = [target_names[i] for i in iris.target]

st.set_page_config(page_title="Iris Flower Classifier", page_icon="🌸", layout="wide")

st.title("🌸 Iris Flower Classification")
st.write("""
This app predicts the species of an Iris flower based on its measurements using a Random Forest model.
""")

st.sidebar.header("Input Features")

def get_user_input():
    col1, col2 = st.sidebar.columns(2)
    with col1:
        sepal_length = st.slider('Sepal length (cm)', 4.0, 8.0, 5.1)
        sepal_width = st.slider('Sepal width (cm)', 2.0, 5.0, 3.5)
    with col2:
        petal_length = st.slider('Petal length (cm)', 1.0, 7.0, 1.4)
        petal_width = st.slider('Petal width (cm)', 0.1, 2.5, 0.2)
    
    user_data = {
        'sepal length (cm)': sepal_length,
        'sepal width (cm)': sepal_width,
        'petal length (cm)': petal_length,
        'petal width (cm)': petal_width
    }
    
    features = pd.DataFrame(user_data, index=[0])
    return features

user_input = get_user_input()

tab1, tab2, tab3 = st.tabs(["Prediction", "Model Explanation", "Data Exploration"])

with tab1:
    st.subheader("Your Input Features")
    st.write(user_input)
    
    model = load_model()
    if model is not None:
        prediction = model.predict(user_input)
        prediction_proba = model.predict_proba(user_input)
        
        st.subheader("Prediction")
        species_emoji = {"setosa": "🌼", "versicolor": "🌺", "virginica": "🏵️"}
        emoji = species_emoji.get(target_names[prediction[0]].lower(), "🌸")
        st.success(f"{emoji} The predicted species is **{target_names[prediction[0]]}**")
        
        st.subheader("Prediction Probabilities")
        proba_df = pd.DataFrame(prediction_proba, columns=target_names)
        
        col1, col2 = st.columns(2)
        with col1:
            st.dataframe(proba_df.style.format("{:.2%}").background_gradient(cmap='Blues'))
        with col2:
            fig, ax = plt.subplots()
            proba_df.T.plot(kind='bar', ax=ax, legend=False)
            ax.set_ylabel("Probability")
            ax.set_title("Class Probabilities")
            ax.set_ylim(0, 1)
            st.pyplot(fig)

with tab2:
    if model is not None:
        st.subheader("SHAP Explanation")
        st.write("""
        SHAP (SHapley Additive exPlanations) values show how each feature contributes to the prediction.
        - Red bars push the prediction towards higher values (virginica)
        - Blue bars push towards lower values (setosa)
        """)
        
        try:
            explainer = shap.TreeExplainer(model.named_steps['classifier'])
            
            scaler = model.named_steps['scaler']
            pca = model.named_steps['pca']
            X_transformed = pca.transform(scaler.transform(user_input))
            
            shap_values = explainer.shap_values(X_transformed)
            
            st.set_option('deprecation.showPyplotGlobalUse', False)
            plt.figure()
            shap.force_plot(explainer.expected_value[0], 
                            shap_values[0], 
                            X_transformed[0],
                            feature_names=["PC1", "PC2"],
                            matplotlib=True,
                            show=False)
            st.pyplot(bbox_inches='tight')
            plt.clf()
            
            st.subheader("Global Feature Importance")
            st.write("This shows which features were most important for the model overall")
            
            importances = model.named_steps['classifier'].feature_importances_
            importance_df = pd.DataFrame({
                'Feature': ["PC1", "PC2"],
                'Importance': importances
            }).sort_values('Importance', ascending=True)
            
            fig, ax = plt.subplots()
            importance_df.plot(kind='barh', x='Feature', y='Importance', ax=ax)
            ax.set_title("PCA Component Importance")
            st.pyplot(fig)
        except Exception as e:
            st.warning(f"Could not generate SHAP explanation: {e}")

with tab3:
    st.subheader("Iris Dataset Exploration")
    
    st.write("""
    ### Pairplot of Features
    The diagonal shows the distribution of each feature, while the other plots show relationships between features.
    """)
    fig = sns.pairplot(iris_df, hue='species', palette='viridis')
    st.pyplot(fig)
    
    st.write("### Feature Distributions by Species")
    feature = st.selectbox("Select feature to visualize", feature_names)
    
    fig, ax = plt.subplots()
    sns.boxplot(x='species', y=feature, data=iris_df, hue='species', palette='viridis', 
                ax=ax, legend=False)
    sns.stripplot(x='species', y=feature, data=iris_df, color='black', alpha=0.5, ax=ax)
    
    if feature in user_input.columns:
        for i, species in enumerate(target_names):
            ax.plot(i, user_input[feature].values[0], 'ro', markersize=10, alpha=0.7)
    
    plt.title(f"Distribution of {feature} by Species")
    st.pyplot(fig)

# Sidebar footer
st.sidebar.markdown("""
---
### About
This app uses a Random Forest classifier trained on the classic Iris dataset.

**Model Details:**
- 100 decision trees
- PCA dimensionality reduction
- StandardScaler normalization

**Note:** If you see version warnings, consider re-training the model with your current scikit-learn version.
""")
