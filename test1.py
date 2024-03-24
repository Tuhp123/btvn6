pip install scikit-learn
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

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Random Forest classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Make predictions on the test set
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
st.write(f"Model Accuracy: {accuracy:.2f}")

def predict_species(sepal_length, sepal_width, petal_length, petal_width):
    # Create a DataFrame with the input features
    data = pd.DataFrame({
        'sepal_length': [sepal_length],
        'sepal_width': [sepal_width],
        'petal_length': [petal_length],
        'petal_width': [petal_width]
    })

    # Make prediction
    prediction = clf.predict(data)
    predicted_species = iris.target_names[prediction[0]]
    return predicted_species

def main():
    st.title("Iris Flower Species Prediction")
    
    # Add inputs for features
    sepal_length = st.slider("Sepal length", 4.0, 8.0, 5.0)
    sepal_width = st.slider("Sepal width", 2.0, 4.5, 3.0)
    petal_length = st.slider("Petal length", 1.0, 7.0, 4.0)
    petal_width = st.slider("Petal width", 0.1, 2.5, 1.0)
    
    # Predict the species
    if st.button("Predict"):
        species_prediction = predict_species(sepal_length, sepal_width, petal_length, petal_width)
        st.write(f"Predicted species: {species_prediction}")

if __name__ == "__main__":
    main()
