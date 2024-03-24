import streamlit as st
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Train a Random Forest classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X, y)

# Sidebar for user input
st.sidebar.header('User Input')
sepal_length = st.sidebar.slider("Sepal length", 4.0, 8.0, 5.0)
sepal_width = st.sidebar.slider("Sepal width", 2.0, 4.5, 3.0)
petal_length = st.sidebar.slider("Petal length", 1.0, 7.0, 4.0)
petal_width = st.sidebar.slider("Petal width", 0.1, 2.5, 1.0)

# Make prediction
def predict_species(sepal_length, sepal_width, petal_length, petal_width):
    data = pd.DataFrame({
        'sepal_length': [sepal_length],
        'sepal_width': [sepal_width],
        'petal_length': [petal_length],
        'petal_width': [petal_width]
    })
    prediction = clf.predict(data)
    return prediction

# Display prediction
st.title("Iris Flower Species Prediction")
st.write("## Input Features")
st.write(f"Sepal length: {sepal_length}")
st.write(f"Sepal width: {sepal_width}")
st.write(f"Petal length: {petal_length}")
st.write(f"Petal width: {petal_width}")

if st.button("Predict"):
    prediction = predict_species(sepal_length, sepal_width, petal_length, petal_width)
    species = iris.target_names[prediction[0]]
    st.write("## Prediction")
    st.write(f"The predicted species is: {species}")
