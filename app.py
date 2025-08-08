import streamlit as st
import pandas as pd
import pickle
import json
import matplotlib.pyplot as plt

# Load model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

# Load dataset
df = pd.read_csv("data/dataset.csv")

# Load model performance
with open("model_performance.json", "r") as f:
    performance = json.load(f)

st.title("🚢 Titanic Survival Prediction App")

st.write("This Streamlit app predicts whether a passenger would survive the Titanic disaster based on their details.")

# Sidebar inputs
st.sidebar.header("Passenger Details")
pclass = st.sidebar.selectbox("Passenger Class (1 = 1st, 2 = 2nd, 3 = 3rd)", [1, 2, 3])
sex = st.sidebar.selectbox("Sex", ["male", "female"])
age = st.sidebar.slider("Age", 0, 80, 25)
sibsp = st.sidebar.slider("Number of Siblings/Spouses Aboard", 0, 8, 0)
parch = st.sidebar.slider("Number of Parents/Children Aboard", 0, 6, 0)
fare = st.sidebar.slider("Fare", 0.0, 500.0, 32.0)

# Prepare input
input_data = pd.DataFrame({
    "Pclass": [pclass],
    "Sex": [1 if sex == "male" else 0],
    "Age": [age],
    "SibSp": [sibsp],
    "Parch": [parch],
    "Fare": [fare]
})

# Prediction
if st.sidebar.button("Predict Survival"):
    prediction = model.predict(input_data)[0]
    result = "Survived 🟢" if prediction == 1 else "Did Not Survive 🔴"
    st.subheader("Prediction Result:")
    st.success(result)

# Show performance metrics
st.subheader("📊 Model Performance")
st.json(performance)

# Show dataset preview
st.subheader("📄 Dataset Preview")
st.dataframe(df.head())

# Show images
st.subheader("📈 Data Visualisations")
st.image("images/age_distribution.png", caption="Age Distribution")
st.image("images/survival_by_sex.png", caption="Survival by Sex")
st.image("images/survival_by_pclass.png", caption="Survival by Passenger Class")
st.image("images/confusion_matrix.png", caption="Confusion Matrix")
