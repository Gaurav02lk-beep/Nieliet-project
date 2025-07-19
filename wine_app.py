import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load data function
@st.cache  # use @st.cache if your Streamlit version <1.18
def load_data():
    data = pd.read_csv('wine_quality.csv')
    return data

df = load_data()

# Prepare data
X = df.drop('quality', axis=1)
y = df['quality'].apply(lambda x: 1 if x >= 7 else 0)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Streamlit UI
st.title("Wine Quality Prediction")

st.sidebar.header("Input Features")

fixed_acidity = st.sidebar.slider('Fixed Acidity', 4.0, 16.0, 7.4)
volatile_acidity = st.sidebar.slider('Volatile Acidity', 0.1, 1.6, 0.7)
citric_acid = st.sidebar.slider('Citric Acid', 0.0, 1.0, 0.0)
residual_sugar = st.sidebar.slider('Residual Sugar', 0.5, 16.0, 1.9)
chlorides = st.sidebar.slider('Chlorides', 0.01, 0.2, 0.076)
free_sulfur = st.sidebar.slider('Free Sulfur Dioxide', 1.0, 72.0, 11.0)
total_sulfur = st.sidebar.slider('Total Sulfur Dioxide', 6.0, 289.0, 34.0)
density = st.sidebar.slider('Density', 0.9900, 1.0050, 0.9978)
pH = st.sidebar.slider('pH', 2.7, 4.0, 3.51)
sulphates = st.sidebar.slider('Sulphates', 0.3, 2.0, 0.56)
alcohol = st.sidebar.slider('Alcohol', 8.0, 15.0, 9.4)

input_data = np.array([[fixed_acidity, volatile_acidity, citric_acid, residual_sugar,
                        chlorides, free_sulfur, total_sulfur, density, pH, sulphates, alcohol]])
input_scaled = scaler.transform(input_data)

if st.button("Predict Wine Quality"):
    prediction = model.predict(input_scaled)[0]
    st.write("Prediction:", "Good Quality Wine 🍷" if prediction == 1 else "Bad Quality Wine 🍶")
