# Titanic Survival Predictor (Streamlit App)

This repository contains a Streamlit application and trained model for predicting Titanic passenger survival.
Files included:
- app.py : Streamlit application
- requirements.txt : Python dependencies
- model.pkl : Trained sklearn pipeline (preprocessing + model)
- data/dataset.csv : The Titanic dataset used for training
- model_performance.json : Cross-validation and test metrics
- age_distribution.png, survival_by_sex.png, survival_by_pclass.png : Visualization images
- confusion_matrix.png : Confusion matrix for best model

## How to run locally
1. Create a virtual environment and install requirements:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```
2. Run the app:
```bash
streamlit run app.py
```

## Deploy to Streamlit Cloud
1. Create a GitHub repository and push all files.
2. Connect the repo to Streamlit Cloud (https://share.streamlit.io).
3. Configure the repo/branch and start the app. The app requires the files in the root (app.py, model.pkl, data/, requirements.txt).

