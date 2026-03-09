✈️ AeroMind AI — Multi-Stage Airline Delay Prediction System
📌 Overview

AeroMind AI is a real-world machine learning system designed to predict airline delays at multiple operational stages.

The system models airline operations from scheduling to arrival using two specialized AI pipelines:

• Stage 1 (Pre-Departure) — Predicts delay risk before takeoff
• Stage 2 (Post-Departure) — Predicts actual arrival delay after takeoff

This project simulates how airlines use predictive analytics for operational planning and disruption management.

⚙️ Key Features

• Multi-stage machine learning system
• Leakage-safe modeling pipeline
• Fold-aware aggregation features
• Time-aware rolling airline performance feature
• Operational recovery insights

📊 Results
Stage 1 — Pre-Departure Model

• Model: Logistic Regression (Fold-Aware)
• ROC-AUC: 0.726

Stage 2 — Post-Departure Model

• Model: Linear Regression
• R² Score: 0.94

🧠 Key Insights

• Airline and airport behavior strongly influence delay risk
• Airlines often recover small delays mid-flight
• Extreme delays remain difficult to predict

🛠 Tech Stack

Python
Pandas
NumPy
Scikit-learn
Matplotlib
Seaborn

📁 Project Structure
src/
   data_ingestion.py
   data_cleaning.py
   feature_engineering.py
   rolling_feature_engineering.py
   model_pre_departure.py
   model_post_departure.py
🚀 Future Improvements

• Real-time prediction pipeline
• Weather data integration
• Deep learning experimentation

👤 Author

Preetham Deepak Kumar
