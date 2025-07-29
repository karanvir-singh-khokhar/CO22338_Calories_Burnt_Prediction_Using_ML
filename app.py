# Import necessary libraries
import streamlit as st
import numpy as np
import pandas as pd
import pickle

# Page Configuration
st.set_page_config(page_title="Calorithm", page_icon="🔥", layout="centered")

# Background image with dark overlay
st.markdown(
    """
    <style>
    .stApp {
        background-image: url("https://static.vecteezy.com/system/resources/thumbnails/037/228/850/small_2x/ai-generated-exercise-machines-in-a-gym-free-photo.jpg");
        background-size: cover;
        background-repeat: no-repeat;
        background-attachment: fixed;
        background-position: center;
    }
    .stApp::before {
        content: "";
        position: fixed;
        top: 0; left: 0; right: 0; bottom: 0;
        background: rgba(0, 0, 0, 0.4);  /* dark overlay for readability */
        z-index: -1;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Load models
models = {
    'Linear Regression': pickle.load(open('lr.pkl', 'rb')),
    'Ridge': pickle.load(open('rd.pkl', 'rb')),
    'Lasso': pickle.load(open('ls.pkl', 'rb')),
    'Decision Tree': pickle.load(open('dtr.pkl', 'rb')),
    'Random Forest': pickle.load(open('rfr.pkl', 'rb'))
}

# Load training data
x_train = pd.read_csv('X_train.csv')

# Model comparison data
comparison_data = {
    'Model': ['Linear Regression', 'Ridge', 'Lasso', 'Decision Tree', 'Random Forest'],
    'MSE': [131.99574, 131.99625, 143.82689, 27.15266, 6.97433],
    'R2_Score': [0.96729, 0.96729, 0.96436, 0.99327, 0.99827]
}
comparison_df = pd.DataFrame(comparison_data)

# Prediction function
def pred(model, Gender, Age, Height, Weight, Duration, Heart_rate, Body_temp):
    Gender = 1 if Gender.lower() == 'male' else 0
    features = np.array([[Gender, Age, Height, Weight, Duration, Heart_rate, Body_temp]])
    prediction = model.predict(features).reshape(1, -1)
    return prediction[0]

# Title and subtitle (centered)
st.markdown("<h1 style='text-align: center;'>🔥 Calorithm 🔥</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align: center;'>🚴‍♂️ Your body's algorithm decoded</h4>", unsafe_allow_html=True)

# Model Comparison Section
with st.expander("📊 Compare Model Performance"):
    st.markdown("<h3 style='text-align: center;'>Model Comparison (MSE and R² Score)</h3>", unsafe_allow_html=True)
    st.dataframe(
        comparison_df.style.highlight_min(subset=['MSE'], color='lightgreen')
                        .highlight_max(subset=['R2_Score'], color='lightblue')
    )
    best_model = comparison_df.sort_values(by=['R2_Score', 'MSE'], ascending=[False, True]).iloc[0]
    st.success(f"🏆 Best Model: **{best_model['Model']}**\n\n(MSE: `{best_model['MSE']:.2f}`, R² Score: `{best_model['R2_Score']:.5f}`)")

# Model Selection
st.markdown("<h3 style='text-align: center;'>🤖 Choose a Machine Learning Model</h3>", unsafe_allow_html=True)
model_choice = st.selectbox('', list(models.keys()))
model = models[model_choice]

# Input Features
st.markdown("<h3 style='text-align: center;'>📥 Enter Input Features</h3>", unsafe_allow_html=True)
col1, col2 = st.columns(2)
with col1:
    Gender = st.selectbox('Gender', ['Male', 'Female'])
    Age = st.selectbox('Age', sorted(x_train['Age'].unique()))
    Height = st.selectbox('Height (cm)', sorted(x_train['Height'].unique()))
    Weight = st.selectbox('Weight (kg)', sorted(x_train['Weight'].unique()))
with col2:
    Duration = st.selectbox('Workout Duration (min)', sorted(x_train['Duration'].unique()))
    Heart_rate = st.selectbox('Heart Rate (bpm)', sorted(x_train['Heart_Rate'].unique()))
    Body_temp = st.selectbox('Body Temperature (°C)', sorted(x_train['Body_Temp'].unique()))

# Predict Button in Center
st.markdown("<br>", unsafe_allow_html=True)
center_col = st.columns(3)[1]
with center_col:
    if st.button('🔍 Predict Calories Burnt'):
        result = pred(model, Gender, Age, Height, Weight, Duration, Heart_rate, Body_temp)
        st.markdown("---")
        st.success("✅ Prediction Complete!")
        st.metric(label="🔥 Calories Burnt", value=f"{result[0]:.2f} kcal")
        st.markdown("---")
