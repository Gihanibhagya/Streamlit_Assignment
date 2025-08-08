import streamlit as st
import pandas as pd
import pickle
import json
import os

# Show current directory files (debug purpose)
st.write("Current directory files:", os.listdir())

# Load model.pkl safely
try:
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)  # or use: model = pickle.load(f, encoding='latin1')
except Exception as e:
    st.error(f"Error loading model.pkl: {e}")
    st.stop()

# Load dataset.csv safely
try:
    df = pd.read_csv("data/dataset.csv")
except Exception as e:
    st.error(f"Error loading dataset.csv: {e}")
    df = None

# Load model performance safely
try:
    with open("model_performance.json", "r") as f:
        performance = json.load(f)
except Exception as e:
    st.error(f"Error loading model_performance.json: {e}")
    performance = None

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

# Prepare input for prediction
input_data = pd.DataFrame({
    "Pclass": [pclass],
    "Sex": [1 if sex == "male" else 0],
    "Age": [age],
    "SibSp": [sibsp],
    "Parch": [parch],
    "Fare": [fare]
})

# Prediction button and logic
if st.sidebar.button("Predict Survival"):
    try:
        prediction = model.predict(input_data)[0]
        result = "Survived 🟢" if prediction == 1 else "Did Not Survive 🔴"
        st.subheader("Prediction Result:")
        st.success(result)
    except Exception as e:
        st.error(f"Error during prediction: {e}")

# Show performance metrics if loaded
if performance:
    st.subheader("📊 Model Performance")
    st.json(performance)

# Show dataset preview if loaded
if df is not None:
    st.subheader("📄 Dataset Preview")
    st.dataframe(df.head())
else:
    st.write("Dataset not loaded, cannot display preview.")

# Show images if files exist
st.subheader("📈 Data Visualisations")

image_files = {
    "Age Distribution": "images/age_distribution.png",
    "Survival by Sex": "images/survival_by_sex.png",
    "Survival by Passenger Class": "images/survival_by_pclass.png",
    "Confusion Matrix": "images/confusion_matrix.png"
}

for caption, path in image_files.items():
    if os.path.exists(path):
        st.image(path, caption=caption)
    else:
        st.write(f"Image not found: {path}")
